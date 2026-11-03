"""WP15 — resolution, confirmation and measurement statistics, third generation.

Covers (package acceptance list):
* the resolvable_diff v2 <-> v1 CONSISTENCY MATRIX (no drift / no confound / no
  new best subset -> the two estimators agree, and the fallback is byte-equal);
* the layered estimator's SYNTHETIC BANDWIDTH COVERAGE (known-sigma paired
  samples: the band covers 3 sigma; a lucky draw does not pin it);
* the LUCKY-SAMPLING BEST fixture (the old behavior is pinned by a lucky draw;
  the new behavior corrects to the confirmation median);
* the confirmation ACCOUNTING on the IterationRecord;
* the NEW-FUNCTION statistics (combined_workload_test / best_standard_error —
  contract C8's new-name rule) and their data-poor fallbacks;
* the drift EXCLUSION with its two-level fallback (plateau / best / sigma);
* the detect_plateau <-> watermark band SYNC (G8.2: one estimator, both judges);
* the top-k numeric FAIL feedback;
* the gate-only confirmation mode's submit assertion (mock backend payload);
* the SPRT win-direction coupling flag plumbing (C14).
"""

from __future__ import annotations

import json
import math
import random

import pytest

from forgekernel.inner.trajectory import IterationRecord, append_record, load_records
from forgekernel.inner.trajectory import build_digest
from forgekernel.loop.parse import (combined_workload_test, detect_plateau,
                                    best_standard_error, parse_eval,
                                    resolvable_diff, _resolvable_diff_v1,
                                    RESOLVABLE_DIFF_CAP, RESOLVABLE_DIFF_FLOOR,
                                    _correctness_error)


def _rec(i, ratio=None, *, drift=False, dev=None, ref=None, confirm=None, wl=None,
         ms=2.0, sha=None):
    kw = dict(device_median_ms=ms, correctness=True, verdict="KEEP",
              decision_id=f"D{i}", change="c", clock_drift=drift)
    if ratio is not None:
        kw["vs_ref_device"] = ratio
    if dev and ref:
        # the trajectory keeps no top-level raw arrays (the WP1 digest-budget
        # decision); the within-job component reads the workload blocks
        wl = list(wl or [])
        wl.append({"name": f"w{i}", "device_readings_ms": list(dev),
                   "ref_device_readings_ms": list(ref)})
    if confirm:
        kw["confirm_readings"] = confirm
    if wl:
        kw["workloads"] = wl
    if sha:
        kw["sha"] = sha
    return IterationRecord(iter=i, **kw)


# --- the v1 <-> v2 consistency matrix (C8: same data, same answer where the old
# estimator was honest; the documented divergences are exactly the biases) -----

def test_consistency_flat_series_both_clamp_to_the_floor():
    recs = [_rec(i, 1.0) for i in range(6)]
    assert resolvable_diff(recs, "device") == RESOLVABLE_DIFF_FLOOR
    assert _resolvable_diff_v1(recs, "device", floor=RESOLVABLE_DIFF_FLOOR,
                               cap=RESOLVABLE_DIFF_CAP) == RESOLVABLE_DIFF_FLOOR


def test_consistency_data_poor_fallback_is_byte_equal():
    assert resolvable_diff([], "device") == RESOLVABLE_DIFF_CAP == 0.015
    assert _resolvable_diff_v1([], "device", floor=RESOLVABLE_DIFF_FLOOR,
                               cap=RESOLVABLE_DIFF_CAP) == RESOLVABLE_DIFF_CAP
    two = [_rec(0, 1.0), _rec(1, 1.001)]
    assert resolvable_diff(two, "device") == 0.015
    assert _resolvable_diff_v1(two, "device", floor=0.005, cap=0.015) == 0.015


def test_consistency_quiet_noise_series_stays_in_family():
    # sub-0.5% paired noise: both estimators clamp to the floor — the v2 layering
    # does not widen a quiet run's band
    rng = random.Random(11)
    recs = [_rec(i, 1.0 * rng.lognormvariate(0.0, 0.001)) for i in range(6)]
    assert resolvable_diff(recs, "device") == RESOLVABLE_DIFF_FLOOR


def test_v2_clamp_discipline_on_every_input_shape():
    # C8's clamp semantics survive the layering: no input shape — flat, trending,
    # outlier-pinned, raw-injected — moves the band outside [floor, cap]
    rng = random.Random(11)
    for recs in (
        [_rec(i, 1.0) for i in range(6)],
        [_rec(i, 1.0 + 0.01 * i) for i in range(6)],
        [_rec(i, 1.0) for i in range(5)] + [_rec(5, 1.05)],
        [_rec(i, 1.0 * rng.lognormvariate(0.0, 0.004)) for i in range(6)],
        [_rec(i, 1.0) for i in range(4)],
    ):
        band = resolvable_diff(recs, "device")
        assert RESOLVABLE_DIFF_FLOOR <= band <= RESOLVABLE_DIFF_CAP
    injected = resolvable_diff([_rec(i, 1.0) for i in range(4)],
                               "device", raw={"ratios": [1.0, 1.02, 0.98, 1.01]})
    assert RESOLVABLE_DIFF_FLOOR <= injected <= RESOLVABLE_DIFF_CAP


def test_v2_no_longer_blunts_on_progress():
    # the v1 self-blunting: a monotone climb reads every old draw's distance-to-
    # best as noise. v2's centered residuals see a wide honest spread either way
    # here (the climb IS wide), but the DIRECTION is pinned by the regression
    # table: v2 <= v1 on the same data, never wider.
    climbs = [1.0, 1.01, 1.02, 1.03, 1.04, 1.05]
    recs = [_rec(i, r) for i, r in enumerate(climbs)]
    assert resolvable_diff(recs, "device") <= _resolvable_diff_v1(
        recs, "device", floor=0.005, cap=0.015)


# --- synthetic bandwidth coverage (known-sigma paired samples) ----------------

def test_band_covers_three_sigma_of_known_window_noise():
    # per-window paired noise of known sigma (2 per record, 5 records): the
    # within-job component must estimate it and the band must clear 3 sigma —
    # while staying below the cap the way a honest interval does
    rng = random.Random(23)
    sigma = 0.004
    recs = []
    for i in range(6):
        ker = [1.0 * rng.lognormvariate(0.0, sigma) for _ in range(4)]
        ref = [1.0 * rng.lognormvariate(0.0, sigma) for _ in range(4)]
        recs.append(_rec(i, sum(ref) / sum(ker), dev=ker, ref=ref))
    band = resolvable_diff(recs, "device")
    # the window-pair deviations ride the ref/kernel lognormal noise (~sigma each
    # side -> ~sigma*sqrt2 on the ratio), deviated around the record's own center:
    # 3 sigma of that, clamped
    assert band >= 3.0 * 0.5 * sigma       # never blind to the noise it saw
    assert band <= RESOLVABLE_DIFF_CAP


def test_injected_confirmation_ratios_tighten_nothing_loosen_nothing_wrongly():
    # C8's raw injection: same-sha confirmation ratios pool into the between-job
    # sigma. A tight confirmation history on a flat run keeps the floor.
    recs = [_rec(i, 1.0) for i in range(5)]
    raw = {"ratios": [1.0, 1.001, 0.999, 1.0005]}
    assert resolvable_diff(recs, "device", raw=raw) == RESOLVABLE_DIFF_FLOOR


# --- the lucky-sampling best fixture ------------------------------------------

def test_lucky_best_is_corrected_to_the_confirmation_median(tmp_path):
    # THE fixture: a lucky single draw (1.976 vs true ~2.0) used to PIN the best
    # (max-of-noise ratchet, docstring's own words). Under WP15 the reading
    # confirms with one independent job and the row banks the median.
    from forgekernel.loop.driver import LoopContext, _confirm_suspicious_watermark

    class _Be:
        def __init__(self, logs):
            self.logs = list(logs)
            self.modes = []

        def submit_gpu(self, branch, mode):
            self.modes.append(mode)
            return self.logs.pop(0)

    eval_log = ('=== JSON ===\n{"correctness": {"pass": true}, "performance": '
                '{"bwd_median_ms": 2.0, "device_median_ms": 2.0}}')
    be = _Be([eval_log])
    ctx = LoopContext(op="demo", workspace=tmp_path, milestone=tmp_path / "m.md",
                      kb_dir=tmp_path, status_path=tmp_path / "status.jsonl",
                      meta_dir=tmp_path / "meta", kernel_file="k.py",
                      branch_prefix="b", target_ms=1.0, deadline_epoch=1e18,
                      backend=be, now=lambda: 0.0)
    ctx.status_path.write_text("", encoding="utf-8")
    ctx.gate_window = "device"
    for i, r in enumerate([1.20, 1.2005, 1.2002, 1.2008, 1.2001, 1.2004]):
        append_record(ctx.status_path, _rec(i, r))
    ev = parse_eval('=== JSON ===\n{"correctness": {"pass": true}, "performance": '
                    '{"bwd_median_ms": 2.4, "device_median_ms": 1.976}}')
    out, info = _confirm_suspicious_watermark(ctx, be, "b", ev, 2.0, profiler={})
    # old behavior: 1.976 stands (the lucky draw IS the bar) — pinned by the
    # consistency table in CHANGES; new behavior: the confirmation median
    assert out.device_median_ms == pytest.approx((1.976 + 2.0) / 2)
    assert be.modes == ["all"]            # the standing budget, one job
    assert info["confirm_jobs"] == 1


def test_confirmation_accounting_rides_the_record(tmp_path):
    from test_loop_driver import FakeBackend, NCU, REC, _ctx

    eval_log = ('=== JSON ===\n{"correctness": {"pass": true}, "performance": '
                '{"bwd_median_ms": 2.4, "device_median_ms": 1.994}}')
    confirm_log = ('=== JSON ===\n{"correctness": {"pass": true}, "performance": '
                   '{"bwd_median_ms": 2.4, "device_median_ms": 2.0}}')
    be = FakeBackend([REC("D1", 1.994)], [eval_log, NCU, confirm_log])
    ctx = _ctx(tmp_path, be)
    ctx.gate_window = "device"
    ctx.watermark_margin = 0.03   # _ctx zeroes it; the A1 routine needs the band
    append_record(ctx.status_path, _rec(0, 1.2, ms=2.0, sha="shabest"))
    rec = __import__("forgekernel.loop.driver", fromlist=["run_inner_iteration"]) \
        .run_inner_iteration(ctx, 1, metric_before=2.0)
    assert rec.confirm_jobs == 1
    assert rec.confirm_readings == [1.994, 2.0]
    assert rec.device_median_ms == pytest.approx(1.997)
    assert rec.best_standard_error is not None and rec.best_standard_error > 0
    (back,) = [r for r in load_records(ctx.status_path) if r.iter == 1]
    assert back.confirm_jobs == 1         # survives the round trip


# --- the new statistics (contract C8: new function names) ---------------------

def test_combined_workload_test_flags_consistent_small_gain():
    # per-record jitter keeps each workload's own sigma alive (identical values
    # would make MAD 0 and every workload unspeakable)
    rng = random.Random(5)
    recs = []
    for i in range(5):
        wl = [{"name": "a", "vs_ref_device": 1.007 * rng.lognormvariate(0.0, 0.001)},
              {"name": "b", "vs_ref_device": 1.006 * rng.lognormvariate(0.0, 0.001)}]
        recs.append(_rec(i, 1.0065, wl=wl))
    out = combined_workload_test(recs, "device")
    assert out is not None and out["significant"] is True and out["z"] > 2.0
    assert {r["name"] for r in out["workloads"]} == {"a", "b"}


def test_combined_workload_test_ignores_a_geomean_cancelled_split():
    # one workload up, one down — the geomean cancels and so must the test
    rng = random.Random(6)
    recs = []
    for i in range(5):
        wl = [{"name": "a", "vs_ref_device": 1.02 * rng.lognormvariate(0.0, 0.001)},
              {"name": "b", "vs_ref_device": 0.99 * rng.lognormvariate(0.0, 0.001)}]
        recs.append(_rec(i, 1.0045, wl=wl))
    out = combined_workload_test(recs, "device")
    assert out is None or out["significant"] is False


def test_combined_workload_test_data_poor_returns_none():
    assert combined_workload_test([], "device") is None
    assert combined_workload_test([_rec(0, 1.0)], "device") is None


def test_best_standard_error_confirms_and_within_paths():
    recs = [_rec(0, 1.0, dev=[1.0, 1.01, 1.005], ref=[1.0, 1.0, 1.0])]
    out = best_standard_error(recs, "device")
    assert out["basis"] == "within_job" and out["n"] == 3
    assert out["ci95"][0] < out["best_ms"] < out["ci95"][1]
    conf = [_rec(1, 1.0, confirm=[0.99, 1.002, 0.998])]
    out2 = best_standard_error(conf, "device")
    assert out2["basis"] == "confirm_jobs" and out2["n"] == 3
    assert out2["ci95"] is not None


def test_best_standard_error_single_draw_is_an_absent_verdict():
    out = best_standard_error([_rec(0, 1.0, ms=2.0)], "device")
    assert out["ci95"] is None and out["se_log"] is None


def test_digest_carries_best_ci95_and_low_confidence():
    recs = [_rec(0, 1.0, confirm=[0.99, 1.002, 0.998])]
    digest = build_digest(recs, {"value": 1.0}, "device")
    ach = digest["achieved"]
    assert ach["best_ci95"] is not None
    assert ach["best_standard_error"] is not None
    assert ach["best_low_confidence"] is False


def test_digest_clock_header_reads_the_last_window(tmp_path):
    gs = {"available": True, "sm_clock_mhz": 2000.0,
          "window_readings": [{"sm_clock_mhz": 2050.0}, {"sm_clock_mhz": 1800.0}]}
    recs = [IterationRecord(iter=3, decision_id="D", change="c", correctness=True,
                            verdict="KEEP", device_median_ms=1.0, gpu_state=gs)]
    digest = build_digest(recs, {"value": 1.0}, "device")
    clock = digest["clock"]
    assert clock["iter"] == 3
    assert clock["sm_clock_mhz"] == 1800.0        # the LAST window, not the mean
    assert clock["last_vs_mean_pct"] < 0          # the downclock the mean hid


# --- drift exclusion, two-level fallback --------------------------------------

def test_drift_rows_drop_from_best_plateau_and_sigma():
    clean = [_rec(i, 1.0, ms=2.0) for i in range(5)]
    drifted_fast = [_rec(9, 1.03, ms=1.94, drift=True)]
    # best: the drifted lucky draw must not become the bar while clean rows speak
    from forgekernel.loop.driver import _best_ms
    assert _best_ms(clean + drifted_fast, "device") == 2.0
    assert _best_ms(drifted_fast * 6, "device") == 1.94   # all-drift: measure as before
    # sigma: the drifted 3% excursion must not widen the band
    band_clean = resolvable_diff(clean, "device")
    band_mixed = resolvable_diff(clean + drifted_fast, "device")
    assert band_mixed == band_clean == RESOLVABLE_DIFF_FLOOR


def test_plateau_excludes_drift_rows_with_fallback():
    # 5 flat rows + 1 drifted lucky row in the window: the plateau judgment must
    # not read the drifted draw as progress
    recs = [_rec(i, 1.0, ms=2.0) for i in range(5)] + [_rec(9, 1.03, ms=1.94, drift=True)]
    assert detect_plateau(recs, window=3, gate_window="device") is True
    all_drift = [_rec(i, 1.0, ms=2.0, drift=True) for i in range(6)]
    assert detect_plateau(all_drift, window=3, gate_window="device") is True  # measures as before


def test_plateau_and_watermark_band_move_together_g82():
    # G8.2's hard obligation: the plateau margin IS the watermark band — both
    # call the same resolvable_diff, so a v2 change moves both. Pinned by
    # computing each and asserting identity on the same records.
    rng = random.Random(31)
    recs = [_rec(i, 1.0 * rng.lognormvariate(0.0, 0.004)) for i in range(6)]
    from forgekernel.loop.driver import LoopContext
    ctx = LoopContext(op="d", workspace=None, milestone=None, kb_dir=None,
                      status_path=None, meta_dir=None, kernel_file="k.py",
                      branch_prefix="b", target_ms=1.0, deadline_epoch=1e18,
                      backend=None, now=lambda: 0.0)
    band = resolvable_diff(recs, "device", floor=ctx.resolvable_floor,
                           cap=ctx.resolvable_cap)
    margin = min(0.015, band)
    assert margin == min(0.015, resolvable_diff(recs, "device",
                                                floor=ctx.resolvable_floor,
                                                cap=ctx.resolvable_cap))


# --- numeric FAIL feedback: top-k worst checks --------------------------------

def test_numeric_fail_feedback_names_the_worst_checks():
    corr = {"pass": False, "error": "workload w0 {'m': 8}: output y: normalized "
            "error 0.5 > tolerance 0.01",
            "checks": [
                {"workload": "w0", "group": "g0", "output": "y", "error": 0.5,
                 "tolerance": 0.01},
                {"workload": "w0", "group": "g1", "output": "y", "error": 0.3,
                 "tolerance": 0.01},
                {"workload": "w0", "group": "g2", "output": "y", "error": 0.2,
                 "tolerance": 0.01},
                {"workload": "w0", "group": "g3", "output": "y", "error": 0.1,
                 "tolerance": 0.01}]}
    text = _correctness_error(corr)
    assert "worst checks" in text
    assert "g0/y" in text and "g1/y" in text and "g2/y" in text
    assert "g3/y" not in text          # top-k = 3, the tail stays out


def test_numeric_fail_feedback_element_seam():
    corr = {"pass": False, "error": "output y wrong",
            "worst_elements": [{"output": "y", "group": "g0", "index": 42,
                                "value": 1.5, "expected": 1.0, "abs_diff": 0.5},
                               {"output": "y", "group": "g1", "index": 7,
                                "value": 0.0, "expected": 1.0, "abs_diff": 1.0}]}
    text = _correctness_error(corr)
    assert "worst elements" in text
    assert "g1/y[7]" in text         # the element locus: group/output[index]
    assert "g0/y[42]" in text


# --- gate-only confirmation mode: the submit assertion ------------------------

def test_confirm_job_mode_reaches_the_submit_payload(tmp_path):
    from forgekernel.loop.driver import LoopContext, _confirm_suspicious_watermark

    class _Be:
        def __init__(self):
            self.modes = []

        def submit_gpu(self, branch, mode):
            self.modes.append(mode)
            return ('=== JSON ===\n{"correctness": {"pass": true}, "performance": '
                    '{"bwd_median_ms": 2.4, "device_median_ms": 2.0}}')

    be = _Be()
    ctx = LoopContext(op="demo", workspace=tmp_path, milestone=tmp_path / "m.md",
                      kb_dir=tmp_path, status_path=tmp_path / "status.jsonl",
                      meta_dir=tmp_path / "meta", kernel_file="k.py",
                      branch_prefix="b", target_ms=1.0, deadline_epoch=1e18,
                      backend=be, now=lambda: 0.0)
    ctx.status_path.write_text("", encoding="utf-8")
    ctx.gate_window = "device"
    ctx.confirm_job_mode = "gate"      # the card layer's cheaper mode
    for i, r in enumerate([1.20, 1.2005, 1.2002, 1.2008, 1.2001, 1.2004]):
        append_record(ctx.status_path, _rec(i, r))
    ev = parse_eval('=== JSON ===\n{"correctness": {"pass": true}, "performance": '
                    '{"bwd_median_ms": 2.4, "device_median_ms": 1.976}}')
    _confirm_suspicious_watermark(ctx, be, "b", ev, 2.0, profiler={})
    assert be.modes == ["gate"]        # the confirmation rides the declared mode


# --- SPRT win-direction coupling (C14) ----------------------------------------

def test_win_stopped_reading_confirms_before_the_watermark(tmp_path):
    from forgekernel.loop.driver import LoopContext, _confirm_suspicious_watermark

    class _Be:
        def __init__(self):
            self.modes = []

        def submit_gpu(self, branch, mode):
            self.modes.append(mode)
            return ('=== JSON ===\n{"correctness": {"pass": true}, "performance": '
                    '{"bwd_median_ms": 2.4, "device_median_ms": 2.0}}')

    be = _Be()
    ctx = LoopContext(op="demo", workspace=tmp_path, milestone=tmp_path / "m.md",
                      kb_dir=tmp_path, status_path=tmp_path / "status.jsonl",
                      meta_dir=tmp_path / "meta", kernel_file="k.py",
                      branch_prefix="b", target_ms=1.0, deadline_epoch=1e18,
                      backend=be, now=lambda: 0.0)
    ctx.status_path.write_text("", encoding="utf-8")
    ctx.gate_window = "device"
    for i, r in enumerate([1.20, 1.2005, 1.2002, 1.2008, 1.2001, 1.2004]):
        append_record(ctx.status_path, _rec(i, r))
    ev = parse_eval('=== JSON ===\n{"correctness": {"pass": true}, "performance": '
                    '{"bwd_median_ms": 2.4, "device_median_ms": 1.994, '
                    '"sprt": {"decided": 2, "direction": "win", "windows": 5}}}')
    out, info = _confirm_suspicious_watermark(ctx, be, "b", ev, 2.0, profiler={})
    assert "early-stopped in the win direction" in info["why"]
    assert be.modes == ["all"]         # confirmed at the standing budget (C14)


def test_loss_stopped_workload_vetoes_the_card_watermark(tmp_path):
    from forgekernel.loop.driver import LoopContext, _confirm_suspicious_watermark

    class _Be:
        def __init__(self):
            self.modes = []

        def submit_gpu(self, branch, mode):
            self.modes.append(mode)
            return ('=== JSON ===\n{"correctness": {"pass": true}, "performance": '
                    '{"bwd_median_ms": 2.4, "device_median_ms": 2.0}}')

    be = _Be()
    ctx = LoopContext(op="demo", workspace=tmp_path, milestone=tmp_path / "m.md",
                      kb_dir=tmp_path, status_path=tmp_path / "status.jsonl",
                      meta_dir=tmp_path / "meta", kernel_file="k.py",
                      branch_prefix="b", target_ms=1.0, deadline_epoch=1e18,
                      backend=be, now=lambda: 0.0)
    ctx.status_path.write_text("", encoding="utf-8")
    ctx.gate_window = "device"
    for i, r in enumerate([1.20, 1.2005, 1.2002, 1.2008, 1.2001, 1.2004]):
        append_record(ctx.status_path, _rec(i, r))
    wl = [{"name": "a", "vs_ref_device": 1.01,
           "sprt": {"decided": 2, "direction": "loss", "windows": 5}},
          {"name": "b", "vs_ref_device": 1.02}]
    ev = parse_eval('=== JSON ===\n{"correctness": {"pass": true}, "performance": '
                    '{"bwd_median_ms": 2.4, "device_median_ms": 1.994, '
                    '"n_workloads": 2, "workloads": []}}')
    ev.workloads = wl
    ev.n_workloads = 2
    out, info = _confirm_suspicious_watermark(ctx, be, "b", ev, 2.0, profiler={})
    assert "loss-stopped" in info["why"]
    assert be.modes == ["all"]         # the veto buys the missing full-window draw
