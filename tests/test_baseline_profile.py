# Copyright 2026 The RSI-Kernel-Tune Authors
# SPDX-License-Identifier: Apache-2.0

"""The baseline profile (``RKT_PROFILE=baseline``) — the audit paper's spine arm.

One switch that turns EVERY fork-added mechanism that can change a measurement
verdict OFF, so a run under the profile emulates upstream-era measurement
behavior (the fork's upstream merge-base, commit 5030503). The expansion
lives in ``loop/driver.py`` (``_apply_baseline_profile``, run LAST in
``run_two_layer``'s resolver chain so its pins stand over the card layer); this
file is its acceptance surface:

(a) profile-off is a no-op — the machinery present but unarmed must leave the
    current defaults byte-identical (the WP11 switch-off record golden,
    ``test_switches_off_matches_baseline_records``, is reused as the reference
    and must keep passing unchanged);
(b) profile-on matches each documented individual switch-off golden — the
    merge-off records golden (WP11), the replay-probe switch-off bench golden
    (WP29), the fresh_recheck child-marker golden with the draw-seed shift
    neutralized (WP9), the screening in-process worker's kill switch (WP17),
    the SPRT env family (WP15), and the suspicious-winner confirm band at a
    zero budget (WP1/WP15);
(c) a self-maintaining coverage audit, mirroring ``test_dataflow_audit.py``'s
    registry style: every ``RKT_*`` switch read under ``src/`` + ``skills/``
    (the loop package plus the seeded job-side tools — the verdict-relevant
    surface) must be pinned by the profile's registry or registered passive
    with a written reason, and every registry line must still name a live
    read. A new verdict-affecting switch that appears without joining the
    profile fails here. Same grep-level boundary as the dataflow audit: reads
    through a shape the scan does not know must be registered manually; the
    stale-entry check keeps the known lines honest.

Switch table (what the profile pins)
------------------------------------
driver-side (LoopContext pins; see ``driver._BASELINE_PROFILE_CTX``):
  merge_eval_ncu = False          WP11 merged eval+NCU job -> the two-job path
  sprt_early_stop = False         WP15 SPRT early stop (the local-delivery
                                  default is ON)
  sprt_win_stop = False           WP15/C14 win-direction early stop
  judge_edge_recheck = False      WP20 edge-band recheck job
  resolvable_floor = resolvable_cap = 0.015
                                  WP1 resolution estimator -> the historical
                                  1.5% constant. Clamp math verified:
                                  min(cap, max(floor, 3*sigma)) with
                                  floor == cap is the constant for every
                                  sample; the data-poor fallback returns cap;
                                  _ncu_regression_band degenerates to
                                  max(_NCU_REGRESSION_BAND, 1 + 0.015) = 1.05,
                                  the upstream constant.
  confirm_budget_per_iter = 0     WP1/WP15 suspicious-winner confirm band: the
                                  band triggers buy nothing and the single
                                  reading stands — upstream never confirmed a
                                  sub-margin jump (the >margin and NCU
                                  triggers keep the upstream two-job protocol
                                  through the same code path).
  probe_manifest = False          WP19 batched probe transport -> per-script
                                  probe jobs (the pre-WP19 transport).
  background_lane = False / background_lane_obj = None
                                  WP19/C15 idle-GPU background measurement lane
                                  (never a gate input by contract; pinned off
                                  anyway, and the lane object run_two_layer
                                  may already have built is torn down).
  precision_contract = "off"      WP22 declared-dtype precision gate.
job-side (exported into this process's environment; locally delivered jobs
inherit them, the RKT_SPRT_EARLY_STOP house pattern; see
``driver._BASELINE_PROFILE_ENV``):
  RKT_FRESH_RECHECK=1 + RKT_FRESH_SEED=0   WP9 fresh_recheck: the child marker
                                  silences the parent's post-window recheck,
                                  and seed 0 keeps the timed input draws at
                                  the upstream seeds (the same marker shifts
                                  them by 7919 otherwise).
  RKT_REPLAY_PROBE = 0            WP29 replay probe (the documented switch).
  RKT_SCREENING_INPROC = 0        WP17 in-process screening worker -> the
                                  per-call subprocess bench (the historical
                                  transport).
  RKT_SPRT_EARLY_STOP / RKT_SPRT_WIN_STOP / RKT_SPRT_WIN_MARGIN /
  RKT_BUILD_CACHE / RKT_BUILD_CACHE_KEY_NARROW / RKT_MEMO_LANE /
  RKT_CUDA_PRECHECK / RKT_SCREENING_CLOCK_GUARD   — removed: default-off fork
                                  opt-ins an inherited environment could arm.

PASSIVE, deliberately untouched (the profile removes verdict power, not
instrumentation): gpu_state sampling, the WP11 wall-clock buckets, WP20's
gate_degraded accounting (``gate_degraded_ratio``), WP20's board-span
report-only rows (``board_span_guard`` — verified never to flip a verdict,
tests/redteam/test_redteam_accounting.py), the bench-side precision
MEASUREMENT (``RKT_PRECISION`` untouched: with the driver contract pinned off,
the precision record is recording only), the kbmenu / ledger bookkeeping.

Verification notes (findings from verifying the prior inventory)
----------------------------------------------------------------
* WP22 precision contract: upstream @5030503 ALREADY resolves the contract to
  strict by default (upstream loop/precision_gate.py resolve_contract).
  Pinning "off" therefore removes one upstream-era enforcement too, not only
  the fork's WP22 declared-dtype conviction; kept per the audit's inventory
  decision (the WP22 machinery rides the same gate and the arm must not be
  able to convict through it).
* WP20 board_span_guard was listed as verdict-affecting in the prior
  inventory; verification shows it is report-only, so it stays ON under the
  passive-telemetry rule (registered passive in the audit below).
* The external correctness monitor (monitor/runtime, RKT_MON_*) is a separate
  command outside the loop's spawn path; the loop's judgment never reads its
  output — nothing to pin.

Known gaps (documented, not forced)
-----------------------------------
* The WP20 bench-side sentinels (stream-span / timer-identity; the proc census
  is accounting-only) have NO switch: card_template/eval/benchkit.py installs
  them unconditionally and a violation convicts through the fresh_recheck fail
  path (``_wp20_sentinel_blocks`` writes result["fresh_recheck"]). Turning
  them off needs an eval-template change — hot bench logic outside this
  profile's composition-time scope. A profile run whose kernel attacks the
  timing stream or the clocks can therefore still REVERT where upstream would
  have kept; the arm description must state this.
* An ssh delivery's submit.toml [env] is beyond the profile's reach (the env
  does not cross the wire; the _wp15 WARN precedent): an ssh card must not
  declare the pinned variables there.
* The MEMO lane (M3) is task-declared (task.toml [protocol] memo_lane) and
  never defaulted into; the profile strips an inherited RKT_MEMO_LANE but does
  not rewrite a MEMO task's own declaration — MEMO tasks are out of the arm's
  population by construction.

Runs on CPU (torch required for the bench-level goldens); no GPU is needed for
any assertion here, and no performance conclusion may be drawn from local runs
(repo red line).
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import re
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")   # the bench-level goldens run real cards on CPU

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
SKILLS = REPO / "skills"

from rsi_kernel_tune.inner.trajectory import IterationRecord, append_record, load_records
from rsi_kernel_tune.loop import driver
from rsi_kernel_tune.loop.parse import parse_eval, resolvable_diff

from test_loop_driver import NCU, REC, EVAL, FakeBackend, _ctx
from test_wp11_throughput import FAIL_EVAL, NCU_CSV, _baseline_records

# --- hygiene -----------------------------------------------------------------

_PROFILE_ENV_KEYS = ("RKT_PROFILE",) + tuple(driver._BASELINE_PROFILE_ENV)


@pytest.fixture(autouse=True)
def _profile_env_hygiene():
    """The profile expansion writes this process's environment; nothing a test
    arms here may leak into another test (the conftest WP15 precedent, widened
    to the whole profile pin set)."""
    yield
    for k in _PROFILE_ENV_KEYS:
        os.environ.pop(k, None)


@pytest.fixture(autouse=True)
def _edge_recheck_state_hygiene():
    """The edge-band recheck's once-per-run state is module-level; every test
    starts from an empty set (the redteam conftest's rule, mirrored here)."""
    driver._JUDGE_EDGE_RECHECKED.clear()
    yield
    driver._JUDGE_EDGE_RECHECKED.clear()


def _apply_profile(ctx):
    """The documented unit seam: the resolver itself, the way
    test_wp20_apply_cfg_resolves_and_protects drives its resolver."""
    driver._apply_baseline_profile(ctx)
    return ctx


# --- (a) profile-off is a no-op -----------------------------------------------


def test_profile_off_requests_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv(driver.RKT_PROFILE_ENV, raising=False)
    ctx = _ctx(tmp_path, FakeBackend([], []))
    assert driver._baseline_profile_requested(ctx) is False


def test_profile_wrong_name_requests_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv(driver.RKT_PROFILE_ENV, "strict")
    ctx = _ctx(tmp_path, FakeBackend([], []))
    assert driver._baseline_profile_requested(ctx) is False


def test_profile_card_layer_table_arms_and_on_false_does_not(tmp_path, monkeypatch):
    monkeypatch.delenv(driver.RKT_PROFILE_ENV, raising=False)
    ctx = _ctx(tmp_path, FakeBackend([], []))
    (tmp_path / "loop.toml").write_text("[profile.baseline]\non = true\n",
                                        encoding="utf-8")
    assert driver._baseline_profile_requested(ctx) is True
    (tmp_path / "loop.toml").write_text("[profile.baseline]\non = false\n",
                                        encoding="utf-8")
    assert driver._baseline_profile_requested(ctx) is False


def test_profile_off_leaves_every_ctx_field_at_its_default(tmp_path, monkeypatch):
    """With the profile unarmed the expansion must not run at all: a ctx that
    went through the (skipped) resolver site equals a fresh LoopContext on
    every field the profile would pin."""
    monkeypatch.delenv(driver.RKT_PROFILE_ENV, raising=False)
    ctx = _ctx(tmp_path, FakeBackend([], []))
    (tmp_path / "fresh").mkdir()
    fresh = _ctx(tmp_path / "fresh", FakeBackend([], []))
    for attr in driver._BASELINE_PROFILE_CTX:
        assert getattr(ctx, attr) == getattr(fresh, attr), attr
    for k in _PROFILE_ENV_KEYS:
        assert k not in os.environ


def test_profile_off_records_match_the_wp11_switch_off_golden(tmp_path):
    """The reused byte-golden: with the profile machinery present but unarmed,
    the WP11 switch-off records are exactly the hand-written golden (this is
    the same assertion test_switches_off_matches_baseline_records makes; run
    here so a profile-off regression cannot hide behind this file's green)."""
    rows = _baseline_records(tmp_path, merge=False, marker=False)
    assert len(rows) == 2
    keep, revert = rows
    assert keep["verdict"] == "KEEP" and keep["correctness"] is True
    assert keep["bwd_median_ms"] == 2.6 and keep["device_median_ms"] == 2.6
    assert keep["profiler"].get("time_ns") == 1000
    assert revert["verdict"] == "REVERT" and revert["correctness"] is False
    assert revert["error"].startswith("correctness FAIL: dq: max_diff=4.5 tol=0.02")


# --- (b) profile-on matches the individual switch-off goldens ------------------


def _profile_records(tmp_path) -> list[dict]:
    """The WP11 golden scenario under the armed profile. The profile pins
    merge_eval_ncu off, so the scripted logs are the two-job path — the same
    script _baseline_records(merge=False) feeds. Every field the WP11 golden
    pins must come out identical: the profile arm emulates the pre-fork
    transport and judgment, not a new reading."""
    tmp_path.mkdir()
    logs = [EVAL(2.6), NCU_CSV, FAIL_EVAL, NCU_CSV]
    be = FakeBackend([REC("D2.3", 2.6), REC("D2.4", 9.9)], logs)
    ctx = _apply_profile(_ctx(tmp_path, be))
    driver.run_inner_iteration(ctx, 0, metric_before=2.6)
    driver.run_inner_iteration(ctx, 1, metric_before=2.6)
    strip = ("agent_turn_s", "eval_job_s", "ncu_job_s", "teacher_s", "queue_wait_s")
    return [{k: v for k, v in r.__dict__.items() if k not in strip}
            for r in load_records(ctx.status_path)]


def test_profile_on_matches_the_switch_off_records_golden(tmp_path):
    (tmp_path / "golden").mkdir()
    rows = _profile_records(tmp_path / "profile")
    golden = _baseline_records(tmp_path / "golden", merge=False, marker=False)
    assert rows == golden
    keep, revert = rows
    assert keep["verdict"] == "KEEP" and keep["bwd_median_ms"] == 2.6
    assert keep["profiler"].get("time_ns") == 1000
    assert revert["verdict"] == "REVERT"
    assert revert["error"].startswith("correctness FAIL: dq: max_diff=4.5 tol=0.02")


def test_profile_pins_the_mixed_transport_and_defense_fields(tmp_path):
    ctx = _apply_profile(_ctx(tmp_path, FakeBackend([], [])))
    assert ctx.merge_eval_ncu is False
    assert ctx.sprt_early_stop is False and ctx.sprt_win_stop is False
    assert ctx.judge_edge_recheck is False
    assert ctx.precision_contract == "off"
    assert ctx.resolvable_floor == 0.015 and ctx.resolvable_cap == 0.015
    assert ctx.confirm_budget_per_iter == 0
    assert ctx.probe_manifest is False
    assert ctx.background_lane is False and ctx.background_lane_obj is None
    # the pin stands over the card layer: an explicit config-armed value the
    # resolvers would have honored is overridden by the arm
    (tmp_path / "armed").mkdir()
    ctx2 = _ctx(tmp_path / "armed", FakeBackend([], []))
    ctx2.merge_eval_ncu = True
    ctx2.sprt_early_stop = True
    ctx2.precision_contract = "strict"
    ctx2.confirm_budget_per_iter = 2
    _apply_profile(ctx2)
    assert ctx2.merge_eval_ncu is False and ctx2.sprt_early_stop is False
    assert ctx2.precision_contract == "off" and ctx2.confirm_budget_per_iter == 0


def test_profile_env_pins_present_and_opt_ins_stripped(tmp_path, monkeypatch):
    # an inherited environment arms fork opt-ins; the profile strips them
    for k in ("RKT_BUILD_CACHE", "RKT_BUILD_CACHE_KEY_NARROW", "RKT_MEMO_LANE",
              "RKT_CUDA_PRECHECK", "RKT_SCREENING_CLOCK_GUARD",
              "RKT_SPRT_EARLY_STOP"):
        monkeypatch.setenv(k, "1")
    ctx = _apply_profile(_ctx(tmp_path, FakeBackend([], [])))
    assert os.environ["RKT_FRESH_RECHECK"] == "1"
    assert os.environ["RKT_FRESH_SEED"] == "0"
    assert os.environ["RKT_REPLAY_PROBE"] == "0"
    assert os.environ["RKT_SCREENING_INPROC"] == "0"
    for k in ("RKT_BUILD_CACHE", "RKT_BUILD_CACHE_KEY_NARROW", "RKT_MEMO_LANE",
              "RKT_CUDA_PRECHECK", "RKT_SCREENING_CLOCK_GUARD",
              "RKT_SPRT_EARLY_STOP", "RKT_SPRT_WIN_STOP", "RKT_SPRT_WIN_MARGIN"):
        assert k not in os.environ, k


def test_profile_sprt_pins_survive_the_local_delivery_flip(tmp_path, monkeypatch):
    """The ORDER regression: _wp11_apply_cfg's local-delivery flip arms
    RKT_SPRT_EARLY_STOP in this process's env (a FakeBackend names no submit
    config -> the delivery reads as "" -> the local arm). The profile resolver
    runs after it and must leave the family absent."""
    monkeypatch.delenv("RKT_SPRT_EARLY_STOP", raising=False)
    ctx = _ctx(tmp_path, FakeBackend([], []))
    driver._wp11_apply_cfg(ctx)          # the flip: the local default arms it
    assert os.environ["RKT_SPRT_EARLY_STOP"] == "1"
    _apply_profile(ctx)
    assert ctx.sprt_early_stop is False and ctx.sprt_win_stop is False
    for k in ("RKT_SPRT_EARLY_STOP", "RKT_SPRT_WIN_STOP", "RKT_SPRT_WIN_MARGIN"):
        assert k not in os.environ, k


# --- WP1: the resolution estimate degenerates to the constant ------------------


def _seed_ratios(ctx, ratios, ms=2.0):
    for i, r in enumerate(ratios):
        append_record(ctx.status_path, IterationRecord(
            iter=i, decision_id=f"D{i}", change="c", correctness=True, verdict="KEEP",
            bwd_median_ms=ms, device_median_ms=ms, vs_ref_device=r))


_QUIET = [1.20, 1.2005, 1.2002, 1.2008, 1.2001, 1.2004]
_NOISY = [1.05, 1.30, 1.12, 1.28, 1.08, 1.25]


@pytest.mark.parametrize("ratios", [_QUIET, _NOISY, [], [1.2], [1.2, 1.21]],
                         ids=["quiet", "noisy", "empty", "one", "two"])
def test_profile_resolution_estimate_is_the_upstream_constant(tmp_path, ratios):
    """floor == cap == 0.015 degenerates the WP1 estimator to the historical
    1.5% constant: the clamp is the constant for every sample, and the
    data-poor fallback (fewer than four ratio samples) returns the cap — the
    same constant. The defaults (floor 0.005) are shown beside it so the pin
    is visibly a pin, not the default."""
    ctx = _apply_profile(_ctx(tmp_path, FakeBackend([], [])))
    _seed_ratios(ctx, ratios)
    records = load_records(ctx.status_path)
    for window in ("device", "e2e"):
        est = resolvable_diff(records, window,
                              floor=ctx.resolvable_floor, cap=ctx.resolvable_cap)
        assert est == pytest.approx(0.015)
    # the run's own default bounds would have resolved finer on the quiet run
    if len(ratios) >= 4:
        default_est = resolvable_diff(records, "device")
        assert default_est <= 0.015


def test_profile_ncu_regression_band_is_the_upstream_constant(tmp_path):
    ctx = _ctx(tmp_path, FakeBackend([], []))
    _seed_ratios(ctx, _QUIET)
    band = driver._ncu_regression_band(load_records(ctx.status_path), "device",
                                       floor=ctx.resolvable_floor,
                                       cap=ctx.resolvable_cap)
    assert band == pytest.approx(driver._NCU_REGRESSION_BAND)   # 1.05, upstream


# --- WP1/WP15: the confirm band at a zero budget buys nothing ------------------


class _GpuLogBackend:
    """Just enough backend for _confirm_suspicious_watermark (the WP1 test's
    double, mirrored)."""

    def __init__(self, logs):
        self.logs = list(logs)
        self.submitted = []

    def submit_gpu(self, branch, mode):
        self.submitted.append(mode)
        return self.logs.pop(0)


_EVAL_DW = lambda e2e, dev: (  # noqa: E731
    '=== JSON ===\n{"correctness": {"pass": true}, "performance": '
    '{"bwd_median_ms": %r, "device_median_ms": %r}}' % (e2e, dev))


def _wm_ctx_profile(tmp_path, be):
    """The WP1 watermark fixture's context (the loop-driver fixture keeps the
    watermark margin at 0 for scripted-log determinism) with the profile
    applied — the exact context a profile run's confirmation gate sees."""
    from rsi_kernel_tune.loop.driver import LoopContext
    ctx = LoopContext(
        op="demo", workspace=tmp_path, milestone=tmp_path / "m.md", kb_dir=tmp_path,
        status_path=tmp_path / "status.jsonl", meta_dir=tmp_path / "meta",
        kernel_file="k.py", branch_prefix="b", target_ms=1.0, deadline_epoch=1e18,
        backend=be, now=lambda: 0.0)
    ctx.status_path.write_text("", encoding="utf-8")
    ctx.gate_window = "device"
    ctx.watermark_margin = 0.03
    return _apply_profile(ctx)


def test_profile_confirm_band_buys_no_job_and_the_reading_stands(tmp_path):
    """The WP1 golden scenario (a 1.2% jump past a quiet run's band) under the
    profile: the band triggers fire their LOG line, buy ZERO confirmation
    jobs, and the single reading stands unchanged — the upstream-era judgment
    of a sub-resolution jump (no confirmation existed there)."""
    be = _GpuLogBackend([_EVAL_DW(2.4, 2.0)])
    ctx = _wm_ctx_profile(tmp_path, be)
    _seed_ratios(ctx, _QUIET)
    ev = parse_eval(_EVAL_DW(2.4, 1.976))
    out, info = driver._confirm_suspicious_watermark(ctx, be, "b", ev, 2.0,
                                                     profiler={})
    assert out.device_median_ms == pytest.approx(1.976)   # unchanged
    assert info["confirm_jobs"] == 0
    assert info["confirm_readings"] == [1.976]
    assert be.submitted == []
    assert info["why"]       # the band fired (its trigger set is fork-added too)


def test_profile_margin_trigger_keeps_the_upstream_two_job_protocol(tmp_path):
    """Above the margin the confirmation is upstream behavior (the >margin
    jump always bought confirm + tie-break): the profile must not have
    touched it."""
    be = _GpuLogBackend([_EVAL_DW(2.4, 1.90), _EVAL_DW(2.4, 1.92)])
    ctx = _wm_ctx_profile(tmp_path, be)
    _seed_ratios(ctx, _QUIET)
    ev = parse_eval(_EVAL_DW(2.4, 1.90))     # 5% > the 3% margin
    out, info = driver._confirm_suspicious_watermark(ctx, be, "b", ev, 2.0,
                                                     profiler={})
    assert be.submitted == ["all"]           # confirm job bought (agreement)
    assert info["confirm_jobs"] == 1


# --- WP20: the edge-band recheck is off; report-only rows stay on --------------


def _edge_fail_eval(margin: float = 0.95) -> str:
    obj = {"correctness": {"pass": False, "status": "INCORRECT_NUMERICAL",
                           "checks": [{"group": "g0", "output": "output",
                                       "error": margin, "tolerance": 1.0}]},
           "performance": {"device_median_ms": 1.0, "bwd_median_ms": 1.0}}
    return "=== JSON ===\n" + json.dumps(obj) + "\n"


def test_profile_edge_band_fail_judges_immediately(tmp_path):
    """The WP20 zero-verdict-control shape (tests/redteam/
    test_redteam_accounting.py) under the profile: a numeric FAIL inside the
    edge band is judged as measured — no recheck job, the bare REVERT."""
    be = FakeBackend([REC("D1", 1.0)], [_edge_fail_eval(0.95), NCU])
    ctx = _apply_profile(_ctx(tmp_path, be))
    rec = driver.run_inner_iteration(ctx, 0, metric_before=3.0)
    assert rec.verdict == "REVERT"
    assert [c[1] for c in be.calls if c[0] == "gpu"] == ["all", "ncu"]
    assert rec.judge_edge_recheck is None


def test_profile_leaves_the_passive_switches_at_their_defaults(tmp_path):
    """Verdict power is what the profile removes, not instrumentation: the
    passive WP20 switches stay at their None sentinels (the config defaults —
    both ON) and the board-span report-only row is still computed under the
    profile."""
    ctx = _apply_profile(_ctx(tmp_path, FakeBackend([], [])))
    assert ctx.board_span_guard is None          # -> the default True applies
    assert ctx.gate_degraded_ratio is None       # -> the default 0.5 applies
    # and the report-only row still rides a wild reference mirror pair:
    wl = {"device_median_ms": 1.0, "bwd_median_ms": 1.0, "n_workloads": 1,
          "workloads": [{"name": "w0", "device_median_ms": 1.0,
                         "device_readings_ms": [1.0, 1.001],
                         "ref_device_readings_ms": [1.0, 1.5]}]}
    obj = {"correctness": {"pass": True}, "performance": wl}
    be = FakeBackend([REC("D1", 1.0)], ["=== JSON ===\n" + json.dumps(obj) + "\n",
                                        NCU])
    (tmp_path / "row").mkdir()
    ctx2 = _apply_profile(_ctx(tmp_path / "row", be))
    rec = driver.run_inner_iteration(ctx2, 0, metric_before=3.0)
    assert rec.verdict == "KEEP"
    assert rec.board_span_unstable is True


# --- WP9/WP29/WP17: the bench-side pins, against the CPU card ------------------

from test_fresh_recheck import HONEST_KERNEL, _card, _run_card  # noqa: E402


def _profile_job_pins() -> dict:
    """The job-side pins the profile exports, as an env overlay for _run_card —
    exactly what a locally delivered job would inherit."""
    return {k: v for k, v in driver._BASELINE_PROFILE_ENV.items() if v is not None}


def test_bench_profile_fresh_recheck_off_matches_its_switch_off_golden(tmp_path,
                                                                       monkeypatch):
    """WP9: the profile's job pins (child marker + seed 0) reproduce the
    documented individual switch-off golden — the marker alone — byte for byte
    on the CPU-exercisable parent path, and the recheck block is never bought:
    with the marker set the parent spawns no recheck child and writes no
    fresh_recheck block (the unit seam; the recheck-ON arm needs a GPU draw —
    the parent bench hard-codes device 'cuda', exactly why
    tests/test_fresh_recheck.py drives the child directly)."""
    card = _card(tmp_path / "golden", HONEST_KERNEL)
    off = _run_card(card, "--mode", "correctness", **_profile_job_pins())
    marker_only = _run_card(card, "--mode", "correctness", RKT_FRESH_RECHECK="1",
                            RKT_FRESH_SEED="0")
    assert off.returncode == marker_only.returncode == 0
    assert off.stdout == marker_only.stdout          # the pin == the documented switch
    assert "[bench] fresh recheck" not in off.stdout
    # unit seam: with the profile's marker the block is None and no child is
    # bought; without it the parent attempts the recheck
    import importlib.util

    from test_fresh_recheck import PROBLEM

    spec = importlib.util.spec_from_file_location(
        "_baseline_profile_benchkit_fr",
        SRC / "rsi_kernel_tune" / "card_template" / "eval" / "benchkit.py")
    bk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bk)
    problem = json.loads(json.dumps(PROBLEM))

    def _must_not_spawn(problem):
        raise AssertionError("the recheck child must not be bought under the profile")

    monkeypatch.setattr(bk, "_fresh_child", _must_not_spawn)
    monkeypatch.setenv("RKT_FRESH_RECHECK", "1")
    assert bk._fresh_recheck_block(problem) is None
    monkeypatch.setenv("RKT_FRESH_RECHECK", "")
    monkeypatch.setattr(bk, "_fresh_child", lambda problem, seed: {"pass": True})
    assert bk._fresh_recheck_block(problem) == {
        "mode": "regen", "layers": [{"pass": True}], "pass": True}


def test_bench_fresh_recheck_draw_seed_shift_is_neutral_under_the_profile():
    """The child marker shifts the timed input draws' seeds (task_inputs adds
    _recheck_seed_shift()); the profile's RKT_FRESH_SEED=0 keeps the draws at
    the upstream seeds — the shift the marker would otherwise introduce is
    exactly neutralized."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_baseline_profile_benchkit",
        SRC / "rsi_kernel_tune" / "card_template" / "eval" / "benchkit.py")
    bk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bk)
    saved = {k: os.environ.get(k) for k in ("RKT_FRESH_RECHECK", "RKT_FRESH_SEED")}
    try:
        os.environ.pop("RKT_FRESH_RECHECK", None)
        os.environ.pop("RKT_FRESH_SEED", None)
        assert bk._recheck_seed_shift() == 0         # the upstream draw
        os.environ["RKT_FRESH_RECHECK"] = "1"
        assert bk._recheck_seed_shift() == 7919      # the marker alone shifts
        os.environ["RKT_FRESH_SEED"] = "0"
        assert bk._recheck_seed_shift() == 0         # the profile's pin: neutral
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_bench_profile_replay_probe_off_matches_its_switch_off_golden(tmp_path):
    """WP29: with the profile's job pins the probe child is never bought —
    byte-identical to the documented RKT_REPLAY_PROBE=0 golden (composed from
    tests/test_replay_probe.py's switch-off tests)."""
    card = _card(tmp_path / "rp", HONEST_KERNEL)
    off = _run_card(card, "--mode", "correctness", RKT_REPLAY_PROBE="0")
    profiled = _run_card(card, "--mode", "correctness", **_profile_job_pins())
    assert off.returncode == profiled.returncode == 0
    assert off.stdout == profiled.stdout
    assert "replay probe" not in profiled.stdout


def test_bench_profile_replay_probe_never_engages(monkeypatch):
    """The parent-side no-child guarantee under the profile env (the replay
    suite's own test shape)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_baseline_profile_benchkit_rp",
        SRC / "rsi_kernel_tune" / "card_template" / "eval" / "benchkit.py")
    bk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bk)
    monkeypatch.setenv("RKT_REPLAY_PROBE", "0")

    def _must_not_spawn(problem):
        raise AssertionError("the probe child must not be bought under the profile")

    monkeypatch.setattr(bk, "_probe_child", _must_not_spawn)
    result = {"correctness": {"pass": True},
              "performance": {"device_median_ms": 0.5}}
    from types import SimpleNamespace
    bk._maybe_replay_probe(result, {}, SimpleNamespace(impl="kernel"),
                           [{"name": "w0", "device_median_ms": 0.5}])
    assert "replay_probe" not in result


def test_profile_screening_inproc_pin_is_the_documented_kill_switch():
    """WP17: the profile's RKT_SCREENING_INPROC value is exactly the value the
    seeded lever-pricing tool reads as the kill switch (the per-call
    subprocess bench, the historical transport)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_baseline_profile_lever_pricing",
        SKILLS / "lever-pricing" / "lever_pricing_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.BENCH_INPROC_ENV == "RKT_SCREENING_INPROC"
    assert mod._inproc_enabled.__name__ == "_inproc_enabled"
    saved = os.environ.get("RKT_SCREENING_INPROC")
    try:
        os.environ["RKT_SCREENING_INPROC"] = _profile_job_pins()["RKT_SCREENING_INPROC"]
        assert mod._inproc_enabled() is False   # the profile kills the worker
    finally:
        if saved is None:
            os.environ.pop("RKT_SCREENING_INPROC", None)
        else:
            os.environ["RKT_SCREENING_INPROC"] = saved


# --- (c) the self-maintaining coverage audit -----------------------------------
#
# The test_dataflow_audit.py style: a mechanical surface scan, two registries
# with written grounds, and interception in BOTH directions — an unregistered
# switch fails, and a registry line whose read has disappeared fails.

_RKT_NAME_RE = re.compile(r"RKT_[A-Z0-9_]+")

# Reviewed-passive RKT_* switches: never pinned, with the reason (anti-formalism:
# every line must carry a reason). Kept beside the profile's own pin registry
# (driver._BASELINE_PROFILE_ENV, whose entries carry their WP in the driver
# comment block and are asserted live below).
PROFILE_ENV_PASSIVE: dict[str, str] = {
    # --- the arm's own switch ---
    "RKT_PROFILE": "the profile arm's own selector; no bench or gate reads it",
    "RKT_PROFILE_ENV": "the profile arm's selector constant's literal; same as above",
    # --- agent / session plumbing (who and how an agent CLI is driven) ---
    "RKT_AGENT_DRIVER": "agent session plumbing; no reading or verdict reads it",
    "RKT_AGENT_SESSION_ID": "agent session plumbing; lifecycle only",
    "RKT_AGENT_SESSION_MODE": "agent session plumbing; lifecycle only",
    # --- build-cache internals: inert once RKT_BUILD_CACHE is absent (pinned) ---
    "RKT_BUILD_CACHE_DIR": "cache store override; inert with RKT_BUILD_CACHE stripped",
    "RKT_BUILD_CACHE_KEY": "cache key echo (gpu_entry internals); inert off",
    "RKT_BUILD_CACHE_ROOT": "cache root echo (gpu_entry internals); inert off",
    "RKT_CACHE_ENTRY": "cache verify result hand-off; inert off",
    "RKT_CACHE_STAGING": "cache promote staging dir; inert off",
    "RKT_KERNEL_BUILD_DIR": "JIT build-dir override (benchkit + the cache block); "
                            "infra: where nvcc writes, never what is built",
    "RKT_TREE_SHA256": "cache key input; inert with the cache stripped",
    # --- declared-measurement stamps and profiling flags ---
    "RKT_COMPUTE_DTYPE": "WP22 declaration stamp; recorded on rows, never a gate "
                         "(the dtype conviction is the precision gate, pinned off)",
    "RKT_CUDA_LINEINFO": "ncu -lineinfo flag; NCU evidence readability only",
    "RKT_SEG": "segment marker label echoed by entry scripts; telemetry labeling",
    # --- defense internals: inert once their mechanism is pinned off ---
    "RKT_CUDA_PRECHECK_TIMEOUT": "budget of the preflight whose arm is stripped",
    "RKT_FRESH_RECHECK_DEVICE": "recheck child device knob; inert with the recheck off",
    "RKT_FRESH_RECHECK_TIMEOUT_S": "recheck child budget; inert with the recheck off",
    "RKT_REPLAY_PROBE_CHILD": "probe child nesting marker; inert with RKT_REPLAY_PROBE=0",
    "RKT_REPLAY_PROBE_DEVICE": "probe child device; inert off",
    "RKT_REPLAY_PROBE_RATIO": "probe threshold; inert off",
    "RKT_REPLAY_PROBE_TIMEOUT_S": "probe child budget; inert off",
    "RKT_EXIT_RC_PREWARM": "build-cache prewarm rc accounting; best-effort, verdict-neutral",
    # --- passive accounting / telemetry ---
    "RKT_GATE_DEGRADED_RATIO": "gate_degraded accounting row override; never a verdict",
    "RKT_JOB_BEGIN": "job lifecycle timestamp; the queue_wait_s telemetry bucket",
    "RKT_QUEUE_T0": "job lifecycle timestamp; the queue_wait_s telemetry bucket",
    "RKT_PROBE_BATCH_SETTLE_S": "probe batching pacing; transport timing only",
    "RKT_MON_HOOK_DIR": "external correctness monitor runtime; a separate command "
                        "the loop never spawns and never reads back",
    "RKT_MON_KERNEL": "external correctness monitor runtime; never in loop jobs",
    "RKT_MON_MAX_CALLS": "external correctness monitor runtime; never in loop jobs",
    "RKT_MON_OUT": "external correctness monitor runtime; never in loop jobs",
    "RKT_MON_PROFILE": "external correctness monitor runtime; never in loop jobs",
    "RKT_MON_ROOT": "external correctness monitor runtime; never in loop jobs",
    # --- measurement recording kept ON on purpose ---
    "RKT_PRECISION": "the precision MEASUREMENT's bench-side kill switch; the "
                     "profile removes the gate's verdict power driver-side "
                     "(precision_contract off) and keeps the recording (passive "
                     "telemetry rule)",
    # --- infra / plumbing ---
    "RKT_DELIVERY": "localenv delivery-kind plumbing",
    "RKT_GPU": "device selection",
    "RKT_PROVIDER_KEY_ENV": "provider credential env indirection",
    "RKT_REF_KEY": "sealed reference key plumbing",
    "RKT_INPUTS_STORE": "WP11 input transport store; hash-verified byte-identical "
                        "restore with full-archive fallback (tree equality pinned)",
    "RKT_INPUTS_SYNC": "WP11 input transport switch; the restored tree is "
                       "byte-identical (pinned by test_wp11_throughput)",
    "RKT_LB_WORKLOAD_SCOPE": "which workload a meta tool reads; scope selection",
    "RKT_RUN_SYNTAX_DOCS": "launch-surface declaration (syntax docs retrieval)",
    "RKT_RUN_TOOLCHAIN_CONFIG": "launch-surface declaration (toolchain config)",
    "RKT_STUDENT_RETRIEVAL": "launch-surface declaration (retrieval permission); "
                             "prompt-side, no measurement semantics",
    "RKT_TOOLCHAIN_CONFIG": "card toolchain declaration (build provenance)",
    # --- goal-ref session plumbing ---
    "RKT_GOALREF_DEADLINE_PASSED": "ref-from-goal session plumbing",
    "RKT_GOALREF_DONE": "ref-from-goal session plumbing",
    "RKT_GOALREF_PENDING": "ref-from-goal session plumbing",
    "RKT_GOALREF_POLL_S": "ref-from-goal session plumbing",
    "RKT_GOALREF_RESULT": "ref-from-goal session plumbing",
    "RKT_GOALREF_WAIT_S": "ref-from-goal session plumbing",
    "RKT_GOAL_DEADLINE": "goal session plumbing",
    "RKT_GOAL_SESSION_PREFIX": "goal session plumbing",
    "RKT_GOAL_TURN_DIR": "goal session plumbing",
    # --- timeline probe geometry (the probe is opt-in; readings are teacher-side
    #     evidence, never a gate) ---
    "RKT_TL_NROW": "timeline probe %clock64 buffer geometry (probe tool parameter)",
    "RKT_TL_NSITE": "timeline probe %clock64 buffer geometry (probe tool parameter)",
    "RKT_TL_NSLOT": "timeline probe %clock64 buffer geometry (probe tool parameter)",
    "RKT_TL_NWARP": "timeline probe %clock64 buffer geometry (probe tool parameter)",
    "RKT_TL_STRIDE": "timeline probe %clock64 buffer geometry (probe tool parameter)",
}

# Config-key registries (the [loop] / [budget] / [target] key surface). Pinned
# keys must equal driver._BASELINE_PROFILE_CTX's documented set; every other
# key read through the known shapes must carry a reason here.
PROFILE_CONFIG_PINS: dict[str, object] = dict(driver._BASELINE_PROFILE_CTX)
PROFILE_CONFIG_PINS.pop("background_lane_obj")   # runtime teardown, not a key

PROFILE_CONFIG_PASSIVE: dict[str, str] = {
    # --- WP20: report-only / accounting (verified verdict-neutral) ---
    "board_span_guard": "report-only row (never flips a verdict — pinned by "
                        "test_board_span_unstable_reported_never_judged); passive "
                        "telemetry stays ON",
    "gate_degraded_ratio": "gate_degraded accounting row; never a verdict",
    # --- prompt-side surfaces (what the agents read; no judgment semantics) ---
    "kb_menu": "retrieval aid in the teacher prompt; no gate reads it",
    "student_digest_binding": "student prompt context shaping; default off = baseline",
    "student_tail_lines": "digest bound; rides student_digest_binding",
    "teacher_context_bound": "teacher prompt context shaping; default off = baseline",
    "teacher_ncu_tail_lines": "NCU view bound; rides teacher_context_bound",
    "teacher_prior_survey": "teacher-side prior supply; brief content only",
    "teacher_prior_art": "teacher-side prior supply; brief content only",
    "carry_dir": "teacher-side prior supply; brief content only",
    # --- routing / budget economy (buys or routes jobs; never re-judges one) ---
    "cert_every_meta_steps": "CANDIDATES certification schedule; routes probe "
                             "rounds, default 0 (off)",
    "kb_cert_batch_size": "certification job batching; job economy",
    "kb_cert_every_meta": "certification schedule; routes the existing probe budget",
    "pattern_search_budget": "WP16 shrink-grid search budget; pressure routing",
    "pattern_search_stages": "WP16 search shape; pressure routing",
    "pool_age_trigger": "WP16 bundle-pricing trigger; pressure routing",
    "reprice_climb_streak": "WP16 reprice gate stand-down; pricing economy",
    "runner_up_pool_k": "WP16 pool bookkeeping; restart targets only",
    "regime_ncu_max": "WP14 scoped NCU evidence cap; meta-side evidence buying",
    "arch_probe_floor_sort": "WP17 advisory pre-rank; never a price, never a verdict",
    "arch_probe_near_diff_reprobe": "WP17 one re-measure; inert without spare budget",
    "arch_probe_ledger": "WP17 bookkeeping ledger; a gate never reads it",
    "m0_probe_rounds": "WP6 ARCH_PROBE job budget; M0 freeze economy",
    "m0_probe_candidates": "WP8 probe bracket width; M0 freeze economy",
    "confirm_job_mode": "confirmation jobs' gpu_entry mode; card-declared transport, "
                        "default 'all' = the historical mode",
    "resume_session": "agent session lifecycle plumbing",
    "stagnation_k": "WP10 pressure trigger; prompt injection only",
    "multi_candidate_max": "WP10 MC-DELIVERY cap; prescreen economy",
    # --- workflow economies present UPSTREAM (@5030503) — not fork-added ---
    "reuse_unchanged_reading": "upstream-era: archives a same-tree gate row with the "
                               "evaluated KEEP's reading; the judgment semantic is "
                               "the tree's own, not a new draw",
    "declared_gate_close": "upstream-era: gate row close-on-KEEP; bookkeeping",
    "watermark_margin": "upstream-era constant (0.03): the >margin confirmation is "
                        "upstream behavior; the fork's band rides "
                        "confirm_budget_per_iter/resolvable_*, which ARE pinned",
    "reanchor_band": "upstream-era phantom-gap re-anchor band",
    "infeasible_min_diagnoses": "upstream-era stop gating",
    "reopen_in_place": "upstream-era reopen workflow",
    "reopen_requires_price": "upstream-era reopen pricing gate",
    "reopen_max_fruitless": "upstream-era reopen cap",
    "reopen_blocked_uncap": "WP5 budget mechanics; cap accounting",
    "stall_pricing": "upstream-era pricing channel",
    "orch_census": "upstream-era census evidence channel",
    "lever_pricing_rounds": "upstream-era pricing economy",
    "lever_pricing_max_levers": "upstream-era pricing economy",
    "reprice_on_new_best": "upstream-era pricing economy",
    "reprice_min_gain_pct": "upstream-era pricing economy",
    "timeline_probe": "upstream-era evidence channel (default off)",
    "meta_format_retry": "WP5 meta bounce retry; billing mechanics",
    "ncu_on_revert": "WP11 evidence economy on an already-REVERT row; the verdict "
                     "is decided before it",
    "meta_hours_per_round": "WP5 budget mechanics",
    "meta_ext_cap": "WP5 budget mechanics",
    "meta_ext_on_headroom": "WP5 budget mechanics",
    "floor_stop": "WP9 stop-side policy (stop threshold), not a reading judgment",
    "floor_stop_margin": "WP9 stop-side policy",
    "floor_margin": "WP9 stop derivation for weak-ref runs",
    "confirm_jobs": "finish-confirmation job count (upstream-era protocol)",
    "recon_op": "recon tree declaration (launch surface)",
    "ref_cmd": "reference measurement command (launch surface)",
    "simulator_dir": "recon channel 4 path (launch surface)",
    # --- launch-surface declarations and tool availability, present UPSTREAM ---
    "gate_window": "upstream-era: which window the gate judges (device|e2e)",
    "language": "upstream-era: task language declaration",
    "kb_arch": "upstream-era: hardware-knowledge-base declaration",
    "oracle": "upstream-era: oracle path declaration",
    "oracle_mode": "upstream-era: source|blackbox declaration",
    "ncu_available": "upstream-era: NCU tool availability declaration",
    "kb_lint_oracle_tokens": "upstream-era: KB lint citation-token policy (bookkeeping)",
    "arch_ref_ms": "upstream-era: arch baseline ms override (launch derivation)",
    "job_env_files": "upstream-era: card files overlaid onto every job (infra)",
    "job_env_preflight": "upstream-era: pre-iteration harness channel preflight "
                         "(a launch gate, not a run-time verdict switch)",
    "ref_in_job": "upstream-era: in-job comparator declaration",
    "ref_price_op": "upstream-era: ref pricing tool resolution (launch surface)",
    "ref_price_app": "upstream-era: ref pricing tool resolution (launch surface)",
    "arch_ref_submit_op": "upstream-era: arch reference submission tree (launch surface)",
    "ref_price_pm_interval": "upstream-era: PM-sampling interval of the frozen "
                             "pricing recipe (a pricing-job parameter)",
    # --- the external correctness monitor (fork-added hook, evidence artifact) ---
    "correctness_monitor": "run-end external monitor policy; writes an evidence "
                           "artifact (correctness_monitor status in the export) — "
                           "the loop's verdicts are already decided",
    "correctness_monitor_profiles": "the monitor's profile set; same evidence artifact",
}


def _strip_comment(text: str) -> str:
    """Comment-stripped text (per line — the dataflow audit's comment rule)."""
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def _rkt_env_surface() -> dict[str, list[str]]:
    """Every RKT_* name read under src/ + skills/ (comment-stripped), with
    file:line for the report."""
    out: dict[str, list[str]] = {}
    for base in (SRC, SKILLS):
        for py in sorted(base.rglob("*.py")):
            rel = py.relative_to(REPO).as_posix()
            for lineno, raw in enumerate(py.read_text(encoding="utf-8").splitlines(), 1):
                code = _strip_comment(raw)
                for m in _RKT_NAME_RE.finditer(code):
                    out.setdefault(m.group(0), []).append(f"{rel}:{lineno}")
    return out


def _config_key_surface() -> dict[str, list[str]]:
    """Every [loop]/[budget]/[target] key read through the known shapes:
    loop_cfg/budget_cfg .get("k") (the driver resolvers), the cli's and the
    precision gate's cfg.get("loop"| "target" | "budget")...get/"k" chains, the
    driver's _int("k")/_flag("k") helper calls (WP17/WP18), and the DEFAULTS
    dicts the dynamic resolvers read (WP16/WP5 — the dataflow audit's tuple
    precedent: a dynamic shape is audited through its constant table)."""
    out: dict[str, list[str]] = {}
    scanned = {
        "src/rsi_kernel_tune/loop/driver.py":
            (SRC / "rsi_kernel_tune" / "loop" / "driver.py").read_text(
                encoding="utf-8").splitlines(),
        "src/rsi_kernel_tune/loop/cli.py":
            (SRC / "rsi_kernel_tune" / "loop" / "cli.py").read_text(
                encoding="utf-8").splitlines(),
        "src/rsi_kernel_tune/loop/precision_gate.py":
            (SRC / "rsi_kernel_tune" / "loop" / "precision_gate.py").read_text(
                encoding="utf-8").splitlines(),
    }
    shapes = (
        re.compile(r"\b(?:loop_cfg|budget_cfg|lc)\.get\(\"([a-z_][a-z0-9_]*)\""),
        re.compile(r"cfg\.get\(\"(?:loop|target|budget)\"(?:, \{\})?\)[^\n]*?\.get\(\"([a-z_][a-z0-9_]*)\""),
        re.compile(r"cfg\.get\(\"(?:loop|target|budget)\"\), \"([a-z_][a-z0-9_]*)\""),
        re.compile(r"\b(?:_int|_flag|_boolean|_num)\(\"([a-z_][a-z0-9_]*)\""),
    )
    for rel, lines in scanned.items():
        for lineno, raw in enumerate(lines, 1):
            code = _strip_comment(raw)
            for rx in shapes:
                for m in rx.finditer(code):
                    key = m.group(1)
                    if key not in ("loop", "target", "budget"):   # table names, not keys
                        out.setdefault(key, []).append(f"{rel}:{lineno}")
    # the dynamic resolvers' constant tables (AST, the _tuple_const precedent)
    driver_src = scanned["src/rsi_kernel_tune/loop/driver.py"]
    tree = ast.parse("\n".join(driver_src))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id in ("_WP16_CFG_DEFAULTS",
                                                     "_WP5_CFG_DEFAULTS")
                for t in node.targets) and isinstance(node.value, ast.Dict):
            rel = "src/rsi_kernel_tune/loop/driver.py"
            for k in node.value.keys:
                if isinstance(k, ast.Constant):
                    out.setdefault(str(k.value), []).append(
                        f"{rel}:{node.lineno}(defaults)")
    return out


def test_env_audit_surface_is_not_vacuous():
    """Extractor sentinel (the dataflow audit's rule): the scan must find the
    known anchors, so a broken scan cannot pass vacuously."""
    surface = _rkt_env_surface()
    for anchor in ("RKT_FRESH_RECHECK", "RKT_REPLAY_PROBE", "RKT_BUILD_CACHE",
                   "RKT_SPRT_EARLY_STOP", "RKT_SCREENING_INPROC", "RKT_MEMO_LANE"):
        assert anchor in surface, anchor
    cfg = _config_key_surface()
    for anchor in ("merge_eval_ncu", "sprt_early_stop", "judge_edge_recheck",
                   "resolvable_floor", "confirm_budget_per_iter",
                   "background_lane", "pattern_search_budget",   # the dynamic shape
                   "kb_cert_batch_size",                          # the _int() shape
                   "precision_contract"):
        assert anchor in cfg, anchor


def test_every_rkt_env_switch_is_pinned_or_registered_passive():
    """Core interception, both directions: a new RKT_* read must join the
    profile's pins or carry a written passive reason, and a registry line
    whose read has disappeared must be deleted (the dataflow audit's stale
    rule)."""
    surface = _rkt_env_surface()
    pins = set(driver._BASELINE_PROFILE_ENV)
    registered = pins | set(PROFILE_ENV_PASSIVE)
    unregistered = sorted(set(surface) - registered)
    assert not unregistered, (
        "RKT_* switch(es) not covered by the baseline profile — either add the "
        "pin to driver._BASELINE_PROFILE_ENV (verdict-affecting) or register it "
        "in PROFILE_ENV_PASSIVE with a reason (verified passive):\n"
        + "\n".join(f"  {k}  e.g. {surface[k][:2]}" for k in unregistered))
    stale = sorted(registered - set(surface))
    assert not stale, (
        "profile/passive registry lines name no live RKT_* read anymore — "
        "delete them:\n  " + ", ".join(stale))


def test_env_passive_registry_lines_carry_reasons():
    thin = [k for k, v in PROFILE_ENV_PASSIVE.items() if not str(v).strip()]
    assert not thin, f"passive registry entries missing reasons: {thin}"


def test_every_config_switch_is_pinned_or_registered_passive():
    surface = _config_key_surface()
    pins = set(PROFILE_CONFIG_PINS)
    registered = pins | set(PROFILE_CONFIG_PASSIVE)
    unregistered = sorted(set(surface) - registered)
    assert not unregistered, (
        "[loop]/[budget]/[target] key(s) not covered by the baseline profile — "
        "pin it in driver._BASELINE_PROFILE_CTX (verdict-affecting) or register "
        "it in PROFILE_CONFIG_PASSIVE with a reason:\n"
        + "\n".join(f"  {k}  e.g. {surface[k][:2]}" for k in unregistered))
    # staleness runs against the BROADER literal surface (any quoted snake_case
    # literal in the scanned files): a tight-shape miss on a hand-written read
    # (a spec projection tuple, a helper-arg forwarding) must not force a
    # registry deletion while the key is still read; a key removed everywhere
    # still loses its literal and is flagged.
    literals = set()
    for path in ("driver.py", "cli.py", "precision_gate.py"):
        text = (SRC / "rsi_kernel_tune" / "loop" / path).read_text(encoding="utf-8")
        literals |= set(re.findall(r"\"([a-z_][a-z0-9_]*)\"", _strip_comment(text)))
    stale = sorted(registered - literals)
    assert not stale, (
        "config registry lines name no live config-key read anymore — delete "
        "them:\n  " + ", ".join(stale))


def test_profile_ctx_pins_match_the_registry_and_the_wiring():
    """The driver-side contract: the assignments _apply_baseline_profile makes
    are exactly the documented registry (a new pin without a registry line
    fails here, and vice versa), and the resolver runs LAST in run_two_layer —
    after every config resolver, so no _wp*_apply_cfg can clobber a pin."""
    fn = driver._apply_baseline_profile
    assigns = {}
    tree = ast.parse(inspect.getsource(fn))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Attribute) \
                and isinstance(node.targets[0].value, ast.Name) \
                and node.targets[0].value.id == "ctx":
            assigns[node.targets[0].attr] = ast.literal_eval(node.value)
    assert assigns == dict(driver._BASELINE_PROFILE_CTX)
    # the values that carry the paper's semantics, pinned literally
    assert assigns["resolvable_floor"] == assigns["resolvable_cap"] == 0.015
    assert assigns["confirm_budget_per_iter"] == 0
    assert assigns["precision_contract"] == "off"
    assert assigns["merge_eval_ncu"] is False
    # wiring: run_two_layer calls the profile after the last _wp* resolver
    rt = ast.parse(inspect.getsource(driver.run_two_layer))
    calls = [n.func.id for n in ast.walk(rt) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)]
    assert "_apply_baseline_profile" in calls
    last_wp = max(calls.index(c) for c in calls if c.endswith("_apply_cfg"))
    assert calls.index("_apply_baseline_profile") > last_wp, \
        "the profile resolver must run after every _wp*_apply_cfg"
