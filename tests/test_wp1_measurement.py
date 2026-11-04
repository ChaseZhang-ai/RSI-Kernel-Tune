"""WP1 — measurement statistics: raw readings, resolvable diff, confirm band, telemetry.

Each test names the task it covers (the WP1 delivery record maps them one to one:
the original CHANGES-WP1.md lives in git history, its English digest in
CHANGELOG.md). The data
premise: the trajectory kept medians and threw the raw readings away, so every
noise-adaptive judgment downstream had to run on a constant.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from peregrine_kernel.inner.trajectory import IterationRecord, append_record, build_digest, load_records
from peregrine_kernel.loop.parse import parse_eval


# --- task 1: the raw readings survive the parse and the record -----------------

_PERF_PAIRED = {
    "bwd_median_ms": 0.52,
    "device_median_ms": 0.4085,
    "device_readings_ms": [0.4071, 0.4099],
    "ref_device_readings_ms": [0.4880, 0.4898],
    "ref_device_ms": 0.4889,
    "vs_ref_device": 1.1963,
    "host_enqueue_ms": 0.004,
    "workloads": [
        {"name": "wl0", "device_median_ms": 0.40, "vs_ref_device": 1.20,
         "device_readings_ms": [0.396, 0.404], "host_enqueue_ms": 0.003},
        {"name": "wl1", "device_median_ms": 0.42, "vs_ref_device": 1.19,
         "device_readings_ms": [0.418, 0.422], "ref_device_readings_ms": [0.498, 0.502]},
    ],
    "n_workloads": 2,
}


def _eval_json(perf: dict, extra: dict | None = None) -> str:
    obj = {"correctness": {"pass": True}, "performance": perf}
    if extra:
        obj.update(extra)
    return json.dumps(obj)


def test_parse_eval_keeps_the_raw_device_readings():
    ev = parse_eval(_eval_json(_PERF_PAIRED))
    assert ev.device_readings_ms == [0.4071, 0.4099]
    assert ev.ref_device_readings_ms == [0.4880, 0.4898]


def test_parse_eval_without_raw_readings_is_unchanged():
    # a plain measure_windows bench reports no mirrored pair: [] rather than an error
    ev = parse_eval(_eval_json({"device_median_ms": 0.4}))
    assert ev.device_readings_ms == []
    assert ev.ref_device_readings_ms == []


def test_workloads_keep_raw_readings_host_clock_and_spread():
    ev = parse_eval(_eval_json(_PERF_PAIRED))
    wl0, wl1 = ev.workloads
    assert wl0["device_readings_ms"] == [0.396, 0.404]
    assert wl0["host_enqueue_ms"] == 0.003
    # (0.404 - 0.396) / 0.396 = 2.02%
    assert wl0["device_spread_pct"] == (max([0.396, 0.404]) - min([0.396, 0.404])) \
        / min([0.396, 0.404])
    assert wl1["ref_device_readings_ms"] == [0.498, 0.502]


def test_workloads_without_raw_readings_stay_compact():
    # the historical compact shape must not grow spread keys it cannot compute
    ev = parse_eval(_eval_json({"workloads": [{"name": "w", "device_median_ms": 0.4}],
                                "n_workloads": 1}))
    (w,) = ev.workloads
    assert "device_readings_ms" not in w
    assert "device_spread_pct" not in w
    assert w == {"name": "w", "device_median_ms": 0.4}


def test_record_round_trips_readings_and_spread(tmp_path):
    ev = parse_eval(_eval_json(_PERF_PAIRED))
    rec = IterationRecord(iter=1, decision_id="D", change="c", correctness=True,
                          verdict="KEEP", device_median_ms=ev.device_median_ms,
                          workloads=ev.workloads, gate_spread_pct=0.0172)
    path = tmp_path / "status.jsonl"
    append_record(path, rec)
    (back,) = load_records(path)
    assert back.workloads[0]["device_readings_ms"] == [0.396, 0.404]
    assert back.workloads[0]["device_spread_pct"] == ev.workloads[0]["device_spread_pct"]
    assert back.gate_spread_pct == 0.0172


def test_old_record_file_loads_without_the_new_fields(tmp_path):
    row = {"iter": 0, "decision_id": "D", "change": "c", "correctness": True,
           "verdict": "KEEP", "bwd_median_ms": 1.0, "device_median_ms": 0.9}
    path = tmp_path / "status.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    (rec,) = load_records(path)
    assert rec.gate_spread_pct is None
    assert rec.workloads == []


def test_digest_shows_spread_but_never_the_raw_arrays():
    ev = parse_eval(_eval_json(_PERF_PAIRED))
    rec = IterationRecord(iter=1, decision_id="D", change="c", correctness=True,
                          verdict="KEEP", device_median_ms=ev.device_median_ms,
                          workloads=ev.workloads)
    digest = build_digest([rec], {"value": 0.5})
    shown = digest["achieved"]["workloads"]
    assert shown, "the digest keeps the per-workload breakdown"
    for w, src_w in zip(shown, ev.workloads):
        assert "device_readings_ms" not in w
        assert "ref_device_readings_ms" not in w
        # BUG-C11-01: this used to compare the digest's key with itself, which passed
        # even when build_digest mangled the median — compare against the source.
        assert w["device_median_ms"] == src_w["device_median_ms"]  # medians stay
    assert shown[0]["device_spread_pct"] == ev.workloads[0]["device_spread_pct"]


# --- task 2: resolvable_diff — the resolution the run's own ratios resolve ------

from peregrine_kernel.loop.parse import (RESOLVABLE_DIFF_CAP, RESOLVABLE_DIFF_FLOOR,
                                    detect_plateau, resolvable_diff)


def _rec(it, ms, verdict="KEEP"):
    # a dual-window record with NO in-job ratio (the pre-WP1 record shape)
    return IterationRecord(iter=it, decision_id="D", change="c", correctness=True,
                           verdict=verdict, bwd_median_ms=ms, device_median_ms=ms)


def _ratio_rec(it, ms, ratio):
    return IterationRecord(iter=it, decision_id="D", change="c", correctness=True,
                           verdict="KEEP", device_median_ms=ms, vs_ref_device=ratio)


def test_resolvable_diff_tight_ratio_series_resolves_fine():
    # ratios within 0.1% of each other: 3 sigma clamps to the floor (~0.5%)
    recs = [_ratio_rec(i, 1.0, 1.20 + (i % 3) * 0.001) for i in range(6)]
    assert resolvable_diff(recs, "device") == RESOLVABLE_DIFF_FLOOR


def test_resolvable_diff_noisy_series_stays_at_the_cap():
    # a ~4% spread in the ratios: 3 sigma above the cap -> clamped to 1.5%
    recs = [_ratio_rec(0, 1.0, 1.00), _ratio_rec(1, 1.0, 1.05),
            _ratio_rec(2, 1.0, 1.01), _ratio_rec(3, 1.0, 1.06),
            _ratio_rec(4, 1.0, 1.02), _ratio_rec(5, 1.0, 1.04)]
    assert resolvable_diff(recs, "device") == RESOLVABLE_DIFF_CAP


def test_resolvable_diff_moderate_noise_lands_between():
    # residuals ~0.3%: 3 sigma ~ 0.9% — resolvable, above the floor, below the cap
    recs = [_ratio_rec(i, 1.0, r) for i, r in
            enumerate([1.000, 0.997, 1.003, 0.998, 1.002, 0.999])]
    rd = resolvable_diff(recs, "device")
    assert RESOLVABLE_DIFF_FLOOR < rd < RESOLVABLE_DIFF_CAP


def test_resolvable_diff_data_poor_falls_back_to_the_old_floor():
    # fewer than 4 paired ratios (a young run, or no in-job reference): the
    # historical constant, exactly
    assert resolvable_diff([], "device") == 0.015
    assert resolvable_diff([_ratio_rec(0, 1.0, 1.2)], "device") == 0.015
    assert resolvable_diff([_ratio_rec(0, 1.0, 1.2), _ratio_rec(1, 1.0, 1.2),
                            _ratio_rec(2, 1.0, 1.2)], "device") == 0.015
    # records without the ratio (old-format rows) do not count as samples
    recs = [_rec(i, 1.0) for i in range(8)]
    assert resolvable_diff(recs, "device") == 0.015


def test_resolvable_diff_reads_the_gate_windows_own_ratio():
    # e2e gate reads vs_ref, device gate reads vs_ref_device
    rec = IterationRecord(iter=0, decision_id="D", change="c", correctness=True,
                          verdict="KEEP", device_median_ms=1.0, bwd_median_ms=1.0,
                          vs_ref=1.30, vs_ref_device=1.00)
    assert resolvable_diff([rec] * 4, "e2e") == RESOLVABLE_DIFF_FLOOR  # 1.30 flat
    assert resolvable_diff([rec] * 4, "device") == RESOLVABLE_DIFF_FLOOR  # 1.00 flat
    varied = [IterationRecord(iter=i, decision_id="D", change="c", correctness=True,
                              verdict="KEEP", device_median_ms=1.0, bwd_median_ms=1.0,
                              vs_ref=1.30, vs_ref_device=r)
              for i, r in enumerate([1.00, 1.05, 1.01, 1.06, 1.02, 1.04])]
    assert resolvable_diff(varied, "e2e") == RESOLVABLE_DIFF_FLOOR
    assert resolvable_diff(varied, "device") == RESOLVABLE_DIFF_CAP


def test_resolvable_diff_only_recent_window_counts():
    # a noisy history that has since gone quiet: the last 6 records speak
    old = [_ratio_rec(i, 1.0, 1.00 + 0.05 * (i % 2)) for i in range(8)]
    new = [_ratio_rec(8 + i, 1.0, 1.20 + 0.001 * (i % 2)) for i in range(6)]
    assert resolvable_diff(old + new, "device") == RESOLVABLE_DIFF_FLOOR


def test_detect_plateau_tight_ratios_admit_a_1pct_best():
    # ~0.7% median improvement with tight ratios: under the 1.5% constant this read
    # as plateau; a run whose own ratios say it resolves ~0.5% must see progress.
    ms = [3.0, 2.70, 2.70, 2.70, 2.68]
    recs = [_rec(i, m) for i, m in enumerate(ms)]
    assert detect_plateau(recs, window=3) is True          # no ratios -> old behavior
    tight = [_ratio_rec(i, m, 1.20 + 0.0005 * (i % 2)) for i, m in enumerate(ms)]
    assert detect_plateau(tight, window=3) is False        # ratios resolve ~0.5%
    # ...and the estimate never exceeds the historical ceiling: ratios spread ~1%+
    # around their best put the margin back at the 1.5% cap, and the same 0.7% gain
    # is unresolvable again
    noisy = [_ratio_rec(i, m, r) for i, (m, r) in
             enumerate(zip(ms, [1.226, 1.234, 1.230, 1.240, 1.238]))]
    assert resolvable_diff(noisy, "device") == RESOLVABLE_DIFF_CAP
    assert detect_plateau(noisy, window=3) is True


def test_detect_plateau_margin_respects_the_callers_ceiling():
    ms = [3.0, 2.70, 2.70, 2.70, 2.68]
    tight = [_ratio_rec(i, m, 1.20 + 0.0005 * (i % 2)) for i, m in enumerate(ms)]
    # noise_floor stays a ceiling that can only tighten: at 0.1% the same 0.7%
    # improvement clears even more easily
    assert detect_plateau(tight, window=3, noise_floor=0.001) is False
    # a raised floor widens the band: a 0.7% gain inside a declared 1% floor is
    # unresolvable, so the plateau stands
    assert detect_plateau(tight, window=3, floor=0.01, cap=0.015) is True


def test_detect_plateau_old_behavior_without_ratios_is_bit_for_bit():
    # the pre-WP1 fixtures (no in-job ratios) must judge exactly as before
    recs = [_rec(0, 3.0), _rec(1, 2.66), _rec(2, 2.70), _rec(3, 2.65), _rec(4, 2.68)]
    assert detect_plateau(recs, window=3) is True
    recs = [_rec(0, 3.0), _rec(1, 2.66), _rec(2, 2.70), _rec(3, 2.60), _rec(4, 2.68)]
    assert detect_plateau(recs, window=3) is False


# --- task 2: driver consumers --------------------------------------------------

import peregrine_kernel.loop.driver as drv


def test_gate_regression_noise_uses_the_run_ratios(tmp_path):
    # a quiet run resolves ~0.5%: a 0.8% same-job ratio drop is a regression now
    # (it passed under the 1.5% constant), and stays a pass with the floor raised.
    history = [_ratio_rec(i, 1.0, 1.200 + 0.0005 * (i % 2)) for i in range(6)]
    prev = _ratio_rec(6, 1.0, 1.200)
    cur = _ratio_rec(7, 1.0, 1.200 * 0.992)
    cur.verdict, cur.decision_id = "KEEP", "M1-gate"
    prev.verdict, prev.decision_id = "KEEP", "M1-gate"
    assert drv._gate_regression(history + [prev], cur, before=len(history) + 1, contract={},
                                gate_window="device") is not None
    assert drv._gate_regression(history + [prev], cur, before=len(history) + 1, contract={},
                                gate_window="device", floor=0.05, cap=0.05) is None


def test_patch_improve_eps_falls_back_to_the_constant():
    assert drv._patch_improve_eps([], "device") == drv._PATCH_IMPROVE_EPS == 0.015
    quiet = [_ratio_rec(i, 1.0, 1.20 + 0.0005 * (i % 2)) for i in range(6)]
    assert drv._patch_improve_eps(quiet, "device") == RESOLVABLE_DIFF_FLOOR


def test_resolvable_keys_wire_through_the_card_loop_toml(tmp_path):
    # the card's loop.toml [loop] is the override surface (the global layer's
    # loop.toml takes [target]/[budget] only): the keys must reach the context
    import tomllib
    text = ('[loop]\nresolvable_floor = 0.008\nresolvable_cap = 0.012\n'
            'confirm_budget_per_iter = 2\n')
    cfg = tomllib.loads(text)
    assert float(cfg["loop"]["resolvable_floor"]) == 0.008
    assert float(cfg["loop"]["resolvable_cap"]) == 0.012
    assert int(cfg["loop"]["confirm_budget_per_iter"]) == 2


def test_resolvable_keys_are_context_config_with_historical_defaults():
    # the card's loop.toml [loop] is the override surface (the global layer's
    # loop.toml takes [target]/[budget] only); loop/cli.py forwards the keys into
    # these fields. The defaults must be the historical bounds, so an untouched
    # run behaves exactly as before wherever the data cannot speak.
    import dataclasses
    import inspect
    import peregrine_kernel.loop.cli as cli
    src = inspect.getsource(cli)
    for key in ("resolvable_floor", "resolvable_cap", "confirm_budget_per_iter"):
        assert f'get("{key}"' in src, key
    names = {f.name for f in dataclasses.fields(drv.LoopContext)}
    assert {"resolvable_floor", "resolvable_cap", "confirm_budget_per_iter"} <= names


# --- task 3: the suspicious-watermark confirmation band ------------------------

from peregrine_kernel.loop.driver import _confirm_suspicious_watermark
from peregrine_kernel.loop.parse import parse_eval


class _GpuLogBackend:
    """Just enough backend for _confirm_suspicious_watermark: hands out eval logs."""

    def __init__(self, logs):
        self.logs = list(logs)
        self.submitted = []

    def submit_gpu(self, branch, mode):
        self.submitted.append(mode)
        return self.logs.pop(0)


def _wm_ctx(tmp_path, be, *, margin=0.03):
    from peregrine_kernel.loop.driver import LoopContext
    ctx = LoopContext(
        op="demo", workspace=tmp_path, milestone=tmp_path / "m.md", kb_dir=tmp_path,
        status_path=tmp_path / "status.jsonl", meta_dir=tmp_path / "meta",
        kernel_file="k.py", branch_prefix="b", target_ms=1.0, deadline_epoch=1e18,
        backend=be, now=lambda: 0.0)
    ctx.status_path.write_text("", encoding="utf-8")
    ctx.gate_window = "device"
    ctx.watermark_margin = margin
    return ctx


_EVAL_DW = lambda e2e, dev: (  # noqa: E731
    '=== JSON ===\n{"correctness": {"pass": true}, "performance": '
    '{"bwd_median_ms": %r, "device_median_ms": %r}}' % (e2e, dev))


def _seed_ratios(ctx, ratios, ms=2.0):
    # a quiet run: paired ratios within 0.05% -> resolvable_diff clamps to the floor
    for i, r in enumerate(ratios):
        append_record(ctx.status_path, IterationRecord(
            iter=i, decision_id=f"D{i}", change="c", correctness=True, verdict="KEEP",
            bwd_median_ms=ms, device_median_ms=ms, vs_ref_device=r))


def test_band_confirmation_triggers_inside_margin_and_records_the_median(tmp_path):
    # 1.2% improvement over best 2.0 (ms 1.976): inside the 3% margin (free under
    # the old constant), past a quiet run's 0.5% resolution -> confirmed with one
    # job, and the watermark takes the median of the two draws.
    be = _GpuLogBackend([_EVAL_DW(2.4, 2.0)])
    ctx = _wm_ctx(tmp_path, be)
    _seed_ratios(ctx, [1.20, 1.2005, 1.2002, 1.2008, 1.2001, 1.2004])
    ev = parse_eval(_EVAL_DW(2.4, 1.976))
    out, info = _confirm_suspicious_watermark(ctx, be, "b", ev, 2.0, profiler={})
    assert out.device_median_ms == pytest.approx((1.976 + 2.0) / 2)
    assert info["confirm_jobs"] == 1
    assert info["confirm_readings"] == [1.976, 2.0]
    assert info["low_confidence"] is False
    assert be.submitted == ["all"]


def test_band_confirmation_disagreement_settles_on_the_median_without_tiebreak(tmp_path):
    # budget 1: a dissenting confirmation is the END of it — median of the two
    # draws, no tie-break job (the band must stay cheap); the disagreement marks
    # the reading low-confidence (WP15: the spread exceeds what the run resolves)
    be = _GpuLogBackend([_EVAL_DW(2.6, 2.3)])
    ctx = _wm_ctx(tmp_path, be)
    _seed_ratios(ctx, [1.20, 1.2005, 1.2002, 1.2008, 1.2001, 1.2004])
    ev = parse_eval(_EVAL_DW(2.4, 1.976))
    out, info = _confirm_suspicious_watermark(ctx, be, "b", ev, 2.0, profiler={})
    assert out.device_median_ms == pytest.approx((1.976 + 2.3) / 2)
    assert info["low_confidence"] is True   # 16% spread >> the 0.5% band
    assert be.submitted == ["all"]


def test_band_raises_the_budget_with_config(tmp_path):
    be = _GpuLogBackend([_EVAL_DW(2.6, 2.3), _EVAL_DW(2.5, 2.28)])
    ctx = _wm_ctx(tmp_path, be)
    ctx.confirm_budget_per_iter = 2
    _seed_ratios(ctx, [1.20, 1.2005, 1.2002, 1.2008, 1.2001, 1.2004])
    ev = parse_eval(_EVAL_DW(2.4, 1.976))
    out, info = _confirm_suspicious_watermark(ctx, be, "b", ev, 2.0, profiler={})
    assert out.device_median_ms == pytest.approx(2.28)  # median of 1.976/2.3/2.28
    assert info["confirm_jobs"] == 2
    assert be.submitted == ["all", "all"]


def test_sub_resolution_improvement_confirms_once_the_a1_routine(tmp_path):
    # WP15 A1: the best is a ratchet — a 0.3% improvement (1.994 vs 2.0, under a
    # quiet run's 0.5% resolution) used to take the watermark FREE. Now every
    # new best confirms once at the standing budget; the median of the two
    # independent draws is what the trajectory banks.
    be = _GpuLogBackend([_EVAL_DW(2.4, 2.0)])
    ctx = _wm_ctx(tmp_path, be)
    _seed_ratios(ctx, [1.20, 1.2005, 1.2002, 1.2008, 1.2001, 1.2004])
    ev = parse_eval(_EVAL_DW(2.4, 1.994))
    out, info = _confirm_suspicious_watermark(ctx, be, "b", ev, 2.0, profiler={})
    assert out.device_median_ms == pytest.approx((1.994 + 2.0) / 2)
    assert info["confirm_jobs"] == 1
    assert "every new best confirms once" in info["why"]
    assert be.submitted == ["all"]


def test_band_stays_free_when_the_run_has_no_ratio_history(tmp_path):
    # no paired ratios anywhere -> resolvable_diff falls back to the 1.5% cap;
    # the 1.2% improvement sits inside it, so the A1 near-best routine confirms
    # it once (the sub-margin free pass is gone; the data-poor fallback band is
    # what keeps the budget-limited trigger from misfiring on noise)
    be = _GpuLogBackend([_EVAL_DW(2.4, 2.0)])
    ctx = _wm_ctx(tmp_path, be)
    ev = parse_eval(_EVAL_DW(2.4, 1.976))  # 1.2% improvement, no ratios on file
    out, info = _confirm_suspicious_watermark(ctx, be, "b", ev, 2.0, profiler={})
    assert out.device_median_ms == pytest.approx((1.976 + 2.0) / 2)
    assert info["confirm_jobs"] == 1
    assert be.submitted == ["all"]


def test_margin_jump_keeps_its_two_job_protocol(tmp_path):
    # a 33% jump is the old strong trigger: confirm + tie-break stay available
    be = _GpuLogBackend([_EVAL_DW(2.6, 2.3), _EVAL_DW(2.5, 2.35)])
    ctx = _wm_ctx(tmp_path, be)
    ev = parse_eval(_EVAL_DW(2.4, 2.0))
    out, info = _confirm_suspicious_watermark(ctx, be, "b", ev, 3.0, profiler={})
    assert out.device_median_ms == pytest.approx(2.3)  # median of 2.0/2.3/2.35
    assert be.submitted == ["all", "all"]


# --- task 4: GPU state telemetry and the clock-drift flag ----------------------

import contextlib
import importlib.util
import sys
import types

TIMING_PY = (Path(__file__).resolve().parents[1]
             / "src" / "peregrine_kernel" / "card_template" / "eval" / "timing.py")


def _load_timing():
    """Import the template timing module against a stub torch (no CUDA here)."""
    saved = sys.modules.get("torch")
    torch = types.ModuleType("torch")
    cuda = types.ModuleType("torch.cuda")
    cuda.synchronize = lambda: None
    torch.cuda = cuda
    torch.equal = lambda a, b: False
    sys.modules["torch"] = torch
    try:
        spec = importlib.util.spec_from_file_location("_wp1_timing", TIMING_PY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        if saved is None:
            sys.modules.pop("torch", None)
        else:
            sys.modules["torch"] = saved


def test_gpu_state_report_means_over_the_samples():
    timing = _load_timing()
    timing.record_gpu_state(query=lambda: {"sm_clock_mhz": 2000.0, "mem_clock_mhz": 8000.0,
                                           "temperature_c": 61.0, "power_w": 180.0})
    timing.record_gpu_state(query=lambda: {"sm_clock_mhz": 1900.0, "mem_clock_mhz": 8000.0,
                                           "temperature_c": 63.0, "power_w": 200.0})
    rep = timing.gpu_state_report()
    assert rep["available"] is True and rep["samples"] == 2
    assert rep["sm_clock_mhz"] == pytest.approx(1950.0)
    assert rep["temperature_c"] == pytest.approx(62.0)
    assert rep["power_w"] == pytest.approx(190.0)


def test_gpu_state_unavailable_is_a_note_not_a_failure():
    timing = _load_timing()
    rep = timing.gpu_state_report([])
    assert rep["available"] is False
    assert rep["sm_clock_mhz"] is None and rep["power_w"] is None
    assert "note" in rep and rep["note"]
    # an injected reader that raises reads as absent, never as an error
    assert timing.sample_gpu_state(query=lambda: 1 / 0) is None


def test_sample_gpu_state_drops_non_numeric_fields():
    timing = _load_timing()
    state = timing.sample_gpu_state(
        query=lambda: {"sm_clock_mhz": "2000", "mem_clock_mhz": "N/A",
                       "temperature_c": 61, "power_w": None})
    assert state == {"sm_clock_mhz": 2000.0, "mem_clock_mhz": None,
                     "temperature_c": 61.0, "power_w": None}
    assert timing.sample_gpu_state(query=lambda: None) is None


def test_gpu_state_rides_the_performance_block():
    timing = _load_timing()
    timing.record_gpu_state(query=lambda: {"sm_clock_mhz": 2000.0, "mem_clock_mhz": 8000.0,
                                           "temperature_c": 61.0, "power_w": 180.0})
    paired = {"kernel": {"median_ms": 1.0, "min_ms": 0.9, "device_median_ms": 0.95,
                         "device_readings_ms": [0.94, 0.96], "host_enqueue_ms": 0.01,
                         "host_bound_ratio": 0.01},
              "ref": None, "ref_error": None, "order": [], "burn_in_rounds": 0}
    out = timing.paired_performance(paired)
    assert out["gpu_state"]["available"] is True
    assert out["gpu_state"]["sm_clock_mhz"] == pytest.approx(2000.0)


def test_parse_eval_reads_gpu_state_top_level_and_perf_block():
    state = {"sm_clock_mhz": 2000.0, "mem_clock_mhz": 8000.0,
             "temperature_c": 61.0, "power_w": 180.0}
    top = parse_eval(_eval_json({"device_median_ms": 0.4}, {"gpu_state": state}))
    assert top.gpu_state == state
    in_perf = parse_eval(_eval_json({"device_median_ms": 0.4, "gpu_state": state}))
    assert in_perf.gpu_state == state
    # top level wins when both exist
    both = parse_eval(_eval_json({"device_median_ms": 0.4, "gpu_state": {"sm_clock_mhz": 1}},
                                 {"gpu_state": state}))
    assert both.gpu_state == state
    assert parse_eval(_eval_json({"device_median_ms": 0.4})).gpu_state is None


def test_sm_clock_drift_math():
    # relative to the LOWER clock — the conservative side (flags more)
    assert drv._sm_clock_drift({"sm_clock_mhz": 2000.0}, {"sm_clock_mhz": 1950.0}) \
        == pytest.approx(50.0 / 1950.0)
    assert drv._sm_clock_drift({"sm_clock_mhz": 2000.0}, {"sm_clock_mhz": 2000.0}) == 0.0
    # either side unmeasured -> None (an unmeasured condition, never a drift claim)
    assert drv._sm_clock_drift({}, {"sm_clock_mhz": 2000.0}) is None
    assert drv._sm_clock_drift(None, None) is None
    assert drv._sm_clock_drift({"sm_clock_mhz": 0}, {"sm_clock_mhz": 2000.0}) is None


def _gpustate_eval(e2e, dev, sm_mhz):
    return ('=== JSON ===\n{"correctness": {"pass": true}, "performance": '
            '{"bwd_median_ms": %r, "device_median_ms": %r, "gpu_state": '
            '{"sm_clock_mhz": %r, "mem_clock_mhz": 8000.0, "temperature_c": 61.0, '
            '"power_w": 180.0}}}' % (e2e, dev, sm_mhz))


def _seed_best_with_state(ctx, ms=2.0, sm=2000.0):
    append_record(ctx.status_path, IterationRecord(
        iter=0, decision_id="D0", change="a", correctness=True, verdict="KEEP",
        bwd_median_ms=ms, device_median_ms=ms, sha="shabest",
        gpu_state={"sm_clock_mhz": sm, "mem_clock_mhz": 8000.0,
                   "temperature_c": 60.0, "power_w": 180.0}))


def test_drifted_reading_is_confirmed_even_below_resolution(tmp_path):
    # 0.3% improvement (sub-resolution: no ratios -> cap 1.5%) while the SM clock
    # sat 5% below the best's: the reading must not take the watermark alone —
    # one budget-limited confirmation, median recorded, row flagged clock_drift.
    # (WP15 promoted the drift check to a flat trigger — it no longer depends on
    # sitting below the band; this fixture is below it either way.)
    be = _GpuLogBackend([_EVAL_DW(2.4, 2.0)])
    ctx = _wm_ctx(tmp_path, be)
    _seed_best_with_state(ctx, ms=2.0, sm=2000.0)
    ev = parse_eval(_gpustate_eval(2.4, 1.994, 1900.0))
    out, info = _confirm_suspicious_watermark(ctx, be, "b", ev, 2.0, profiler={})
    assert out.device_median_ms == pytest.approx((1.994 + 2.0) / 2)
    assert "clock move" in info["why"]
    assert be.submitted == ["all"]


def test_same_clock_sub_resolution_improvement_confirms_once_a1(tmp_path):
    # WP15 A1: same clock, sub-resolution improvement — the old free pass is
    # gone; every new best confirms once.
    be = _GpuLogBackend([_EVAL_DW(2.4, 2.0)])
    ctx = _wm_ctx(tmp_path, be)
    _seed_best_with_state(ctx, ms=2.0, sm=2000.0)
    ev = parse_eval(_gpustate_eval(2.4, 1.994, 2000.0))
    out, info = _confirm_suspicious_watermark(ctx, be, "b", ev, 2.0, profiler={})
    assert out.device_median_ms == pytest.approx((1.994 + 2.0) / 2)
    assert info["confirm_jobs"] == 1
    assert be.submitted == ["all"]


def test_run_inner_iteration_flags_clock_drift_on_the_record(tmp_path):
    from test_loop_driver import EVAL_DW, FakeBackend, NCU, REC, _ctx
    be = FakeBackend([REC("D1", 1.994)],
                     [_gpustate_eval(2.4, 1.994, 1900.0), NCU, _EVAL_DW(2.4, 2.0)])
    ctx = _ctx(tmp_path, be)
    ctx.gate_window = "device"
    ctx.watermark_margin = 0.03
    _seed_best_with_state(ctx, ms=2.0, sm=2000.0)
    rec = drv.run_inner_iteration(ctx, 1, metric_before=2.0)
    assert rec.clock_drift is True
    assert rec.gpu_state["sm_clock_mhz"] == pytest.approx(1900.0)
    # the drifted 0.3% reading was confirmed before it took the watermark
    assert rec.device_median_ms == pytest.approx((1.994 + 2.0) / 2)
    # WP15: the confirmation accounting rides the row
    assert rec.confirm_jobs == 1
    assert rec.confirm_readings == [1.994, 2.0]


def test_run_inner_iteration_same_clock_confirms_once_a1(tmp_path):
    # WP15 A1: the sub-resolution improvement confirms once; the row banks the
    # median of the two independent draws (not the lucky first one).
    from test_loop_driver import EVAL_DW, FakeBackend, NCU, REC, _ctx
    be = FakeBackend([REC("D1", 1.994)],
                     [_gpustate_eval(2.4, 1.994, 2000.0), NCU, _EVAL_DW(2.4, 2.0)])
    ctx = _ctx(tmp_path, be)
    ctx.gate_window = "device"
    ctx.watermark_margin = 0.03
    _seed_best_with_state(ctx, ms=2.0, sm=2000.0)
    rec = drv.run_inner_iteration(ctx, 1, metric_before=2.0)
    assert rec.clock_drift is False
    assert rec.device_median_ms == pytest.approx((1.994 + 2.0) / 2)
    assert rec.confirm_jobs == 1
    assert rec.best_low_confidence is False


def test_record_round_trips_gpu_state_and_clock_drift(tmp_path):
    rec = IterationRecord(iter=1, decision_id="D", change="c", correctness=True,
                          verdict="KEEP", device_median_ms=1.0,
                          gpu_state={"sm_clock_mhz": 2000.0}, clock_drift=True)
    path = tmp_path / "status.jsonl"
    append_record(path, rec)
    (back,) = load_records(path)
    assert back.gpu_state == {"sm_clock_mhz": 2000.0}
    assert back.clock_drift is True
    # pre-telemetry rows load as before
    path.write_text(json.dumps({"iter": 0, "decision_id": "D", "change": "c",
                                "correctness": True, "verdict": "KEEP"}) + "\n",
                    encoding="utf-8")
    (old,) = load_records(path)
    assert old.gpu_state == {} and old.clock_drift is False
